import os
from pathlib import Path
from collections import Counter

from fire import Fire
from tqdm import tqdm

from data_loading import select_data
from demonstrations import select_demonstration
from inference import VLLMModel


def make_output_name(**kwargs) -> str:
    parts = [f"{k}={str(v).replace('/', '-')}" for k, v in kwargs.items()]
    return "-".join(parts)


def evaluate(data_name: str, demo_name: str, output_dir: str = "outputs", **kwargs):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    path_out = Path(
        output_dir, make_output_name(eval_data=data_name, demo=demo_name, **kwargs)
    ).with_suffix(".jsonl")
    model = VLLMModel(**kwargs)
    data = select_data(data_name, data_split="test")
    demo = select_demonstration(demo_name)
    model.stopping_words = demo.get_stopping_words()

    scores = []
    progress = tqdm(data.samples, desc=str(path_out))
    path_out.parent.mkdir(parents=True, exist_ok=True)
    with open(path_out, "w") as f:
        for i, sample in enumerate(progress):
            sample.prompt = demo.make_prompt(sample.question)
            sample.raw_outputs.append(model.run(sample.prompt))
            for o in sample.raw_outputs:
                sample.preds.append(demo.extract_answer(o))

            sample.accept_answer = demo.extract_answer(sample.accept_answer)
            scores.append(sample.preds[0] == sample.accept_answer)
            progress.set_postfix(accuracy=sum(scores) / len(scores))
            print(sample.model_dump_json(indent=2))
            print(sample.model_dump_json(), file=f)
            print(dict(sample=i, average_accuracy=sum(scores) / len(scores)))
            
    return sum(scores) / len(scores)


def evaluate_batched(
    data_name: str,
    demo_name: str,
    output_dir: str = "outputs_batched",
    batch_size: int = 32,
    **kwargs,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    path_out = Path(
        output_dir, make_output_name(eval_data=data_name, demo=demo_name, **kwargs)
    ).with_suffix(".jsonl")
    model = VLLMModel(**kwargs)
    data = select_data(data_name, data_split="test")
    demo = select_demonstration(demo_name)
    model.stopping_words = demo.get_stopping_words()

    scores = []
    progress = tqdm(range(0, len(data.samples), batch_size), desc=str(path_out))
    path_out.parent.mkdir(parents=True, exist_ok=True)
    with open(path_out, "w") as f:
        for i in progress:
            batch = data.samples[i : i + batch_size]
            for sample in batch:
                sample.prompt = demo.make_prompt(sample.question)

            outputs = model.run_batch([s.prompt for s in batch])
            for j, sample in enumerate(batch):
                sample.raw_outputs.append(outputs[j])
                for o in sample.raw_outputs:
                    sample.preds.append(demo.extract_answer(o))

                sample.accept_answer = demo.extract_answer(sample.accept_answer)
                scores.append(sample.preds[0] == sample.accept_answer)
                progress.set_postfix(accuracy=sum(scores) / len(scores))
                print(sample.model_dump_json(indent=2))
                print(sample.model_dump_json(), file=f)
                print(dict(sample=i, average_accuracy=sum(scores) / len(scores)))
                
    return sum(scores) / len(scores)


def get_most_common(values: list):
    return Counter(values).most_common()[0][0]


def evaluate_sc(
    data_name: str,
    demo_name: str,
    output_dir: str = "outputs_sc",
    num_sample: int = 10,
    data_split: str = "test",
    **kwargs,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    path_out = Path(
        output_dir,
        make_output_name(
            eval_data=data_name,
            demo=demo_name,
            split=data_split,
            num_sample=num_sample,
            **kwargs,
        ),
    ).with_suffix(".jsonl")

    model = VLLMModel(**kwargs)
    data = select_data(data_name, data_split=data_split)
    demo = select_demonstration(demo_name)
    model.stopping_words = demo.get_stopping_words()

    scores = []
    progress = tqdm(data.samples, desc=str(path_out))
    path_out.parent.mkdir(parents=True, exist_ok=True)
    with open(path_out, "w") as f:
        for i, sample in enumerate(progress):
            sample.prompt = demo.make_prompt(sample.question)
            sample.raw_outputs = model.run_many(sample.prompt, num_sample)
            for o in sample.raw_outputs:
                sample.preds.append(demo.extract_answer(o))

            sample.accept_answer = demo.extract_answer(sample.accept_answer)
            scores.append(get_most_common(sample.preds) == sample.accept_answer)
            progress.set_postfix(accuracy=sum(scores) / len(scores))
            print(sample.model_dump_json(indent=2))
            print(sample.model_dump_json(), file=f)
            print(dict(sample=i, average_accuracy=sum(scores) / len(scores)))
            
    return sum(scores) / len(scores)
            
            
def run_eval_many(*paths: str, **kwargs):
    records = []
    for p in tqdm(paths):
        try:
            score = evaluate_batched(path_model=p, **kwargs)
            # clear_cuda()
        except Exception as e:
            print(e)
            score = -1
        records.append(dict(path=p, score=score))
        for rec in records:
            print(rec)


if __name__ == "__main__":
    Fire()
