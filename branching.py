import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import List
from zipfile import ZipFile

import nltk
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from data_loading import select_data, ReasonData, DataInfo
from demonstrations import select_demonstration
from inference import VLLMModel
from evaluation import make_output_name


class ReasonStepSplitter(BaseModel):
    source_path: str = "punkt.zip"
    extract_path: str = "~/nltk_data/tokenizers"

    def load(self):
        folder = Path(self.extract_path).expanduser()
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            with ZipFile(self.source_path) as f:
                f.extractall(folder)

            try:
                nltk.sent_tokenize("")
                print("NLTK loaded successfully")
            except LookupError:
                print("NLTK not loaded properly, downloading now")
                nltk.download("punkt")

    def run(self, text: str) -> List[str]:
        self.load()
        outputs = []
        for part in text.split("\n"):
            if part.strip():
                for sent in nltk.sent_tokenize(part.strip()):
                    if sent.strip():
                        outputs.append(sent.strip())
        return outputs


def generate_paths(
    data_name: str,
    demo_name: str,
    path_out: str,
    num_sample: int = 10,
    data_split: str = "test",
    start_index: int = None,
    end_index: int = None,
    existing_preds_path: str = "",
    **kwargs,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = VLLMModel(**kwargs)
    data = select_data(data_name, data_split=data_split)
    demo = select_demonstration(demo_name)
    model.stopping_words = demo.get_stopping_words()
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)

    splitter = ReasonStepSplitter()
    print(dict(splitter=type(splitter).__name__))
    random.seed(0)

    if existing_preds_path:
        data = ReasonData.load(existing_preds_path)
        for s in data.samples:
            lst = [o for o, p in zip(s.raw_outputs, s.preds) if p == s.accept_answer]
            s.accept_explanation = "" if not lst else lst[0]
        data.samples = [s for s in data.samples if s.accept_explanation]
        print(dict(existing_preds_path=existing_preds_path, data=len(data.samples)))
    else:
        existing_preds_path = Path(
            "outputs_sc",
            make_output_name(
                eval_data=data_name,
                demo=demo_name,
                split=data_split,
                num_sample=num_sample,
                **kwargs,
            ),
        ).with_suffix(".jsonl")

    if start_index is not None and end_index is not None:
        data.samples = data.samples[start_index:end_index]
        print(dict(start=start_index, end=end_index, samples=len(data.samples)))

    with open(path_out, "w") as f:
        for i, sample in enumerate(tqdm(data.samples, desc=str(path_out))):
            steps = splitter.run(sample.accept_explanation)[:-1]
            prefixes = [""] + ["\n".join(steps[:j]) for j in range(1, len(steps) + 1)]
            print(dict(sample=i, prefixes=prefixes))

            for prefix in prefixes:
                if prefix:
                    sample.answer_prefix = " " + prefix + "\n"
                else:
                    assert sample.answer_prefix == ""

                sample.prompt = demo.make_prompt(sample.question) + sample.answer_prefix
                sample.raw_outputs = model.run_many(sample.prompt, num_sample)
                sample.preds = []
                for o in sample.raw_outputs:
                    sample.preds.append(demo.extract_answer(o))
                print(sample.model_dump_json(indent=2))
                print(sample.model_dump_json(), file=f)


def merge_path_data(*paths: str, path_out: str):
    samples = []
    for p in paths:
        data = ReasonData.load(p)
        samples.extend(data.samples)

    data = ReasonData(samples=samples)
    data.save(path_out)
    print(dict(path_out=path_out, data=len(data.samples)))


def save_tuning_data(
    path_in: str,
    path_out: str,
    use_gold_only: bool = False,
    use_single_pair_only: bool = False,
    path_info: str = "dataset_info.json",
):
    outputs = []
    counts = []
    data = ReasonData.load(path_in)
    seen = set()

    for sample in tqdm(data.samples):
        gold = sample.accept_answer
        pool = sample.raw_outputs
        assert len(sample.raw_outputs) == len(sample.preds)
        counts.append(len(sample.raw_outputs))
        accepts = [o for i, o in enumerate(pool) if sample.preds[i] == gold]
        rejects = [o for i, o in enumerate(pool) if sample.preds[i] != gold]

        if use_gold_only:
            if accepts == [] or rejects == [] or sample.question in seen:
                continue
            seen.add(sample.question)
            prefix = " " if rejects and rejects[0].startswith(" ") else ""
            a = f"{prefix}{sample.accept_explanation.strip('.')}. So the answer is \\boxed{{{sample.accept_answer}}}."
            b = rejects[0]
            raw = dict(
                instruction=sample.question,
                input="",
                output=[a, sample.answer_prefix + b],
            )
            outputs.append(raw)
            continue

        if use_single_pair_only:
            if accepts == [] or rejects == [] or sample.question in seen:
                continue
            seen.add(sample.question)
            raw = dict(
                instruction=sample.question,
                input="",
                output=[
                    sample.answer_prefix + accepts[0],
                    sample.answer_prefix + rejects[0],
                ],
            )
            outputs.append(raw)
            continue

        for a in sorted(set(accepts)):
            for b in sorted(set(rejects)):
                raw = dict(
                    instruction=sample.question,
                    input="",
                    output=[sample.answer_prefix + a, sample.answer_prefix + b],
                )
                outputs.append(raw)

    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    with open(path_out, "w") as f:
        json.dump(outputs, f, indent=2)

    info = DataInfo.load(path_info)
    info.add_new_data(
        Path(path_out).stem,
        dict(file_name=path_out, ranking=True, formatting="alpaca"),
    )

    if use_gold_only or use_single_pair_only:
        questions = [raw["instruction"] for raw in outputs]
        if len(questions) != len(set(questions)):
            breakpoint()
            raise ValueError

    info.save(path_info)
    print("Printing output samples")
    random.seed(0)
    for raw in random.sample(outputs, k=10):
        print(json.dumps(raw, indent=2))
    print(dict(path_out=path_out, samples=len(outputs), num_outputs=Counter(counts)))
    print(dict(orig_questions=len(set(s.question for s in data.samples))))
    print(dict(filtered_questions=len(set(raw["instruction"] for raw in outputs))))


"""
python branching.py generate_paths gsm8k gsm8k outputs/gsm8k_llama3_8b_part1.json --path_model models/Meta-Llama-3-8B --data_split train --start_index 0 --end_index 2000 --existing_preds_path outputs_sc/eval_data=gsm8k-demo=gsm8k-split=train-num_sample=10-path_model=models-Meta-Llama-3-8B.jsonl
python branching.py generate_paths gsm8k gsm8k outputs/gsm8k_llama3_8b_part2.json --path_model models/Meta-Llama-3-8B --data_split train --start_index 2000 --end_index 4000 --existing_preds_path outputs_sc/eval_data=gsm8k-demo=gsm8k-split=train-num_sample=10-path_model=models-Meta-Llama-3-8B.jsonl
python branching.py generate_paths gsm8k gsm8k outputs/gsm8k_llama3_8b_part3.json --path_model models/Meta-Llama-3-8B --data_split train --start_index 4000 --end_index 6000 --existing_preds_path outputs_sc/eval_data=gsm8k-demo=gsm8k-split=train-num_sample=10-path_model=models-Meta-Llama-3-8B.jsonl
python branching.py generate_paths gsm8k gsm8k outputs/gsm8k_llama3_8b_part4.json --path_model models/Meta-Llama-3-8B --data_split train --start_index 6000 --end_index 8000 --existing_preds_path outputs_sc/eval_data=gsm8k-demo=gsm8k-split=train-num_sample=10-path_model=models-Meta-Llama-3-8B.jsonl
python branching.py merge_path_data outputs/gsm8k_llama3_8b_part*.json --path_out outputs/gsm8k_llama3_8b_merged.json

python branching.py generate_paths math math outputs/math_llama3_8b_part1.json --path_model models/Meta-Llama-3-8B --data_split train --start_index 0 --end_index 2000 --existing_preds_path outputs_sc/eval_data=math-demo=math-split=train-num_sample=10-path_model=models-Meta-Llama-3-8B.jsonl
python branching.py generate_paths math math outputs/math_llama3_8b_part2.json --path_model models/Meta-Llama-3-8B --data_split train --start_index 2000 --end_index 4000 --existing_preds_path outputs_sc/eval_data=math-demo=math-split=train-num_sample=10-path_model=models-Meta-Llama-3-8B.jsonl
python branching.py generate_paths math math outputs/math_llama3_8b_part3.json --path_model models/Meta-Llama-3-8B --data_split train --start_index 4000 --end_index 6000 --existing_preds_path outputs_sc/eval_data=math-demo=math-split=train-num_sample=10-path_model=models-Meta-Llama-3-8B.jsonl
python branching.py generate_paths math math outputs/math_llama3_8b_part4.json --path_model models/Meta-Llama-3-8B --data_split train --start_index 6000 --end_index 8000 --existing_preds_path outputs_sc/eval_data=math-demo=math-split=train-num_sample=10-path_model=models-Meta-Llama-3-8B.jsonl
python branching.py merge_path_data outputs/math_llama3_8b_part*.json --path_out outputs/math_llama3_8b_merged.json


"""


if __name__ == "__main__":
    Fire()
