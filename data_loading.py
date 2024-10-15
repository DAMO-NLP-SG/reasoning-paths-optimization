import json
import random
import re
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset
from fire import Fire
from pydantic import BaseModel

from utils import find_math_answer


class ReasonSample(BaseModel):
    question: str
    accept_explanation: str = ""
    accept_answer: str = ""
    prompt: str = ""
    answer_prefix: str = ""
    raw_outputs: List[str] = []
    preds: List[str] = []
    info: dict = {}


class ReasonData(BaseModel):
    samples: List[ReasonSample]

    def analyze(self, seed: int = 0):
        random.seed(seed)
        for sample in random.sample(self.samples, k=10):
            print(sample.model_dump_json(indent=2))
        info = dict(samples=len(self.samples))
        print(json.dumps(info, indent=2))

    @classmethod
    def load(cls, path: str):
        samples = []
        with open(path) as f:
            for line in f:
                samples.append(ReasonSample(**json.loads(line)))
        print(dict(path=path, samples=len(samples)))
        return cls(samples=samples)

    @classmethod
    def load_from_huggingface(cls, data_split: str):
        raise NotImplementedError

    def save(self, path: str):
        with open(path, "w") as f:
            for sample in self.samples:
                print(sample.model_dump_json(), file=f)
        print(dict(path=path, samples=len(self.samples)))


class GSM8KData(ReasonData):
    @classmethod
    def load_from_huggingface(cls, data_split: str):
        samples = []
        for raw in load_dataset("gsm8k", "main", split=data_split):
            explanation, answer = raw["answer"].split("####")
            explanation = re.sub(r"<<[^>]+>>", "", explanation)  # Remove angle brackets
            samples.append(
                ReasonSample(
                    question=raw["question"].strip(),
                    accept_explanation=explanation.strip(),
                    accept_answer=answer.strip(),
                )
            )

        return cls(samples=samples)


class MATHData(ReasonData):
    @classmethod
    def load_from_huggingface(cls, data_split: str):
        samples = []
        for raw in load_dataset("competition_math", "rb", split=data_split):
            answer = find_math_answer(raw["solution"])
            samples.append(
                ReasonSample(
                    question=raw["problem"].strip(),
                    accept_explanation=raw["solution"].strip(),
                    accept_answer=answer.strip(),
                )
            )

        return cls(samples=samples)


class MMLUData(ReasonData):
    @classmethod
    def load_from_huggingface(cls, data_split: str):
        samples = []
        for raw in load_dataset(
            "chiayewken/mmlu",
            split="train+validation" if data_split == "train" else "test",
        ):
            options = [raw["A"], raw["B"], raw["C"], raw["D"]]
            i = "ABCD".index(raw["target"])
            template = "({}) {}"
            answer = template.format(raw["target"], options[i]).split()[0]
            suffix = "\n".join(
                [template.format("ABCD"[j], o) for j, o in enumerate(options)]
            )
            assert answer in suffix

            samples.append(
                ReasonSample(
                    question=raw["input"].strip() + "\n" + suffix,
                    accept_explanation="",
                    accept_answer=answer,
                )
            )

        return cls(samples=samples)


class MMLUStemData(ReasonData):
    @classmethod
    def get_subsets(cls) -> List[str]:
        return [
            "abstract_algebra",
            "astronomy",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "electrical_engineering",
            "elementary_mathematics",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_mathematics",
            "high_school_physics",
            "high_school_statistics",
            "machine_learning",
        ]

    @classmethod
    def load_from_huggingface(cls, data_split: str):
        subsets = set(cls.get_subsets())
        samples = []
        for raw in load_dataset(
            "chiayewken/mmlu",
            split="test" if data_split == "train" else "train+validation",
        ):
            if raw["subset"] not in subsets:
                continue

            options = [raw["A"], raw["B"], raw["C"], raw["D"]]
            i = "ABCD".index(raw["target"])
            template = "({}) {}"
            answer = template.format(raw["target"], options[i]).split()[0]
            suffix = "\n".join(
                [template.format("ABCD"[j], o) for j, o in enumerate(options)]
            )
            assert answer in suffix

            samples.append(
                ReasonSample(
                    question=raw["input"].strip() + "\n" + suffix,
                    accept_explanation="",
                    accept_answer=answer,
                )
            )

        return cls(samples=samples)


class CSQAData(ReasonData):
    @classmethod
    def load_from_huggingface(cls, data_split: str):
        samples = []
        for raw in load_dataset(
            "tau/commonsense_qa",
            split="validation" if data_split == "test" else "train",
        ):
            options = raw["choices"]["text"]
            labels = "".join(raw["choices"]["label"])
            i = labels.index(raw["answerKey"])
            template = "({}) {}"
            answer = template.format(raw["answerKey"], options[i]).split()[0]
            suffix = "\n".join(
                [template.format(labels[j], o) for j, o in enumerate(options)]
            )
            assert answer in suffix

            samples.append(
                ReasonSample(
                    question=raw["question"].strip() + "\n" + suffix,
                    accept_explanation="",
                    accept_answer=answer,
                )
            )

        return cls(samples=samples)


class WINOGRANDEData(ReasonData):
    @classmethod
    def load_from_huggingface(cls, data_split: str):
        samples = []
        for raw in load_dataset(
            "allenai/winogrande",
            name="winogrande_debiased",
            split="validation" if data_split == "test" else "train",
        ):
            options = [raw["option1"], raw["option2"]]
            labels = "12"
            i = labels.index(raw["answer"])
            template = "({}) {}"
            answer = template.format(raw["answer"], options[i]).split()[0]
            suffix = "\n".join(
                [template.format(labels[j], o) for j, o in enumerate(options)]
            )
            assert answer in suffix

            samples.append(
                ReasonSample(
                    question=raw["sentence"].strip() + "\n" + suffix,
                    accept_explanation="",
                    accept_answer=answer,
                )
            )

        return cls(samples=samples)


def select_data(name: str, **kwargs):
    if name == "gsm8k":
        return GSM8KData.load_from_huggingface(**kwargs)
    elif name == "math":
        return MATHData.load_from_huggingface(**kwargs)
    elif name == "mmlu":
        return MMLUData.load_from_huggingface(**kwargs)
    elif name == "mmlu_stem":
        return MMLUStemData.load_from_huggingface(**kwargs)
    elif name == "csqa":
        return CSQAData.load_from_huggingface(**kwargs)
    elif name == "winogrande":
        return WINOGRANDEData.load_from_huggingface(**kwargs)
    else:
        raise KeyError(name)


def test_data(name: str, **kwargs):
    data = select_data(name, **kwargs)
    data.analyze()


class DataInfo(BaseModel):
    info: Dict[str, dict]

    @classmethod
    def load(cls, path: str):
        if not Path(path).exists():
            print("New DataInfo: {}")
            return cls(info={})
        with open(path) as f:
            info = json.load(f)
        return cls(info=info)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(self.info, f, indent=2)
        print(json.dumps(self.info, indent=2))

    def add_new_data(self, name: str, info: dict):
        self.info[name] = info


if __name__ == "__main__":
    Fire()
