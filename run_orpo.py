import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Tuple, Literal, Any

import torch

# noinspection PyPep8Naming
import torch.nn.functional as F
from datasets import Dataset
from llmtuner.data import PairwiseDataCollatorWithPadding, get_dataset, split_dataset
from llmtuner.extras.callbacks import LogCallback
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.ploting import plot_loss
from llmtuner.hparams import (
    ModelArguments,
    get_train_args,
    FinetuningArguments,
    DataArguments,
)
from llmtuner.model import load_model, load_tokenizer
from llmtuner.train.utils import create_custom_optimzer, create_custom_scheduler
from transformers import (
    Trainer,
    PreTrainedModel,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from trl import DPOTrainer
from trl.trainer.utils import disable_dropout_in_model


class CustomORPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        finetuning_args: FinetuningArguments,
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)

        self.finetuning_args = finetuning_args
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.beta = finetuning_args.orpo_beta
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        Trainer.__init__(self, model=model, **kwargs)
        print("Orpo new CustomORPOTrainer")

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            # noinspection PyTypeChecker
            self.optimizer = create_custom_optimzer(
                self.model, self.args, self.finetuning_args
            )
        return super().create_optimizer()

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        # noinspection PyTypeChecker
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @staticmethod
    def odds_ratio_loss(
        chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor"
    ) -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps))
            - torch.log1p(-torch.exp(rejected_logps))
        )
        odds_ratio_loss = -F.logsigmoid(log_odds)
        return odds_ratio_loss

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the average log probabilities of the labels under the given logits.
        """
        all_logits: "torch.Tensor" = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            use_cache=False,
        ).logits.to(torch.float32)

        # noinspection PyTypeChecker
        all_logps = self.get_batch_logps(
            logits=all_logits,
            labels=batch["labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def decode_sequences(self, x: torch.Tensor) -> List[str]:
        sequences = [[i for i in lst if i >= 0] for lst in x.tolist()]
        return self.tokenizer.batch_decode(sequences)

    @staticmethod
    def check_valid(labels) -> bool:
        for i, lst in enumerate(labels.tolist()):
            if len(set(lst)) == 1:
                print(f"Invalid label row at index {i}")
                return False
        return True

    def get_prefix_removed_logps(self, chosen_logits, rejected_logits, labels):
        batch_size = labels.size(0) // 2
        chosen_labels, rejected_labels = labels.clone().split(batch_size, dim=0)
        assert chosen_labels.shape == rejected_labels.shape

        # For a chosen and rejected rationale, we want to ignore the front part that is the same
        # So we create a mask that denotes the longest common prefix
        # Then the labels based on the mask will cause the logp computation to ignore those positions
        matches = torch.eq(chosen_labels, rejected_labels)
        mask = torch.cumprod(matches, dim=1).bool()
        chosen_labels = torch.where(mask, self.label_pad_token_id, chosen_labels)
        rejected_labels = torch.where(mask, self.label_pad_token_id, rejected_labels)
        if not self.check_valid(chosen_labels) or not self.check_valid(rejected_labels):
            breakpoint()

        chosen_logps = self.get_batch_logps(
            logits=chosen_logits,
            labels=chosen_labels,
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        rejected_logps = self.get_batch_logps(
            logits=rejected_logits,
            labels=rejected_labels,
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        if torch.isnan(chosen_logps).any() or torch.isnan(rejected_logps).any():
            breakpoint()
        return chosen_logps, rejected_logps

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the ORPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        chosen_logps, rejected_logps, chosen_logits, rejected_logits = (
            self.concatenated_forward(model, batch)
        )
        sft_loss = -chosen_logps
        chosen_logps, rejected_logps = self.get_prefix_removed_logps(
            chosen_logits, rejected_logits, batch["labels"]
        )
        odds_ratio_loss = self.odds_ratio_loss(chosen_logps, rejected_logps)
        batch_loss = (sft_loss + self.beta * odds_ratio_loss).mean()

        chosen_rewards = self.beta * chosen_logps.detach()
        rejected_rewards = self.beta * rejected_logps.detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.cpu().mean()
        metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.cpu().mean()
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.cpu().mean()
        metrics["{}rewards/margins".format(prefix)] = (
            (chosen_rewards - rejected_rewards).cpu().mean()
        )
        metrics["{}logps/rejected".format(prefix)] = (
            rejected_logps.detach().cpu().mean()
        )
        metrics["{}logps/chosen".format(prefix)] = chosen_logps.detach().cpu().mean()
        metrics["{}logits/rejected".format(prefix)] = (
            rejected_logits.detach().cpu().mean()
        )
        metrics["{}logits/chosen".format(prefix)] = chosen_logits.detach().cpu().mean()
        metrics["{}sft_loss".format(prefix)] = sft_loss.detach().cpu().mean()
        metrics["{}odds_ratio_loss".format(prefix)] = (
            odds_ratio_loss.detach().cpu().mean()
        )

        return batch_loss, metrics


def count_prefix_overlap(lst_a: List[int], lst_b: List[int]) -> int:
    count = 0
    for a, b in zip(lst_a, lst_b):
        if a == b:
            count += 1
        else:
            break
    return count


@dataclass
class ReasonPathsCollator(PairwiseDataCollatorWithPadding):
    dataset: Dataset = None
    sample_groups: Dict[str, List[dict]] = None
    is_seeded: bool = False
    max_group_size: int = 8
    max_length: int = round(1024 * 0.9)

    def load(self):
        if not self.is_seeded:
            random.seed(0)
            self.is_seeded = True

        if self.sample_groups is None:
            assert self.dataset is not None
            assert self.tokenizer is not None
            self.sample_groups = {}

            for i in range(len(self.dataset)):
                sample = self.dataset[i]
                key = str(sample["prompt_ids"])
                self.sample_groups.setdefault(key, []).append(sample)

            print(dict(ReasonPathsCollator_sample_groups=len(self.sample_groups)))
            for key in random.sample(self.sample_groups.keys(), k=4):
                for raw in self.sample_groups[key][:4]:
                    info = dict(
                        prompt=self.tokenizer.decode(raw["prompt_ids"]),
                        chosen=self.tokenizer.decode(raw["chosen_ids"]),
                        reject=self.tokenizer.decode(raw["rejected_ids"]),
                        prefix_overlap=count_prefix_overlap(
                            raw["chosen_ids"], raw["rejected_ids"]
                        ),
                    )
                    print(json.dumps(info, indent=2))
                print("#" * 80)
                group_sizes = [len(lst) for lst in self.sample_groups.values()]
                print(dict(min_group_size=min(group_sizes), max=max(group_sizes)))
            self.test_filter_by_length()

    def count_num_training_steps(self, args: Seq2SeqTrainingArguments) -> int:
        self.load()
        num_questions = len(self.sample_groups)
        batch_size = args.train_batch_size * args.gradient_accumulation_steps
        batch_size *= args.world_size
        return math.ceil(num_questions * args.num_train_epochs / batch_size)

    def test_filter_by_length(self):
        groups = list(self.sample_groups.values())
        info = dict(
            max_length=self.max_length,
            original=sum(len(lst) for lst in groups),
            filtered=sum(len(self.filter_by_length(lst)) for lst in groups),
            original_qns=len(groups),
            filtered_qns=len([lst for lst in groups if self.filter_by_length(lst)]),
        )
        print(json.dumps(info, indent=2))

    def filter_by_length(self, samples: List[dict]) -> List[dict]:
        outputs = []
        for raw in samples:
            chosen = raw["prompt_ids"] + raw["chosen_ids"]
            reject = raw["prompt_ids"] + raw["rejected_ids"]
            if len(chosen) < self.max_length and len(reject) < self.max_length:
                outputs.append(raw)
        return outputs

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        self.load()
        assert features is not None
        del features

        while True:
            # Avoid bias from large groups, which will have more raw samples
            key = random.choice(sorted(self.sample_groups.keys()))

            group = self.filter_by_length(self.sample_groups[key])
            if not group:
                print(f"Empty group after filter: {self.tokenizer.decode(eval(key))}")
                continue

            if len(group) > self.max_group_size:
                group = random.sample(group, k=self.max_group_size)
            return super().__call__(group)


def run_train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
    callbacks: Optional[List[TrainerCallback]] = None,
):
    print(locals())
    if data_args.dataset.startswith("math"):
        data_args.cutoff_len = 1536  # MATH has longer solutions
    tokenizer = load_tokenizer(model_args)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="rm")
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = ReasonPathsCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=(
            IGNORE_INDEX
            if data_args.ignore_pad_token_for_loss
            else tokenizer.pad_token_id
        ),
        dataset=dataset,
        max_length=round(data_args.cutoff_len * 0.9),
    )

    # Update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset
    training_args.max_steps = data_collator.count_num_training_steps(training_args)

    # Initialize our Trainer
    trainer = CustomORPOTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args),
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()
        # noinspection PyArgumentList
        trainer.log_metrics("train", train_result.metrics)
        # noinspection PyArgumentList
        trainer.save_metrics("train", train_result.metrics)
        # noinspection PyArgumentList
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            # noinspection PyTypeChecker
            plot_loss(
                training_args.output_dir,
                keys=["loss", "eval_loss", "rewards/accuracies", "sft_loss"],
            )


def main(
    args: Optional[Dict[str, Any]] = None,
    callbacks: Optional[List[TrainerCallback]] = None,
):
    print("Orpo new main")
    model_args, data_args, training_args, finetuning_args, generating_args = (
        get_train_args(args)
    )
    callbacks = [LogCallback()] if callbacks is None else callbacks
    run_train(model_args, data_args, training_args, finetuning_args, callbacks)


if __name__ == "__main__":
    main()
