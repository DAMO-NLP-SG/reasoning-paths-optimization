from typing import Optional, List

from fire import Fire
from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedTokenizer


class DummyImport:
    LLM = None
    SamplingParams = None


try:
    import vllm
    from vllm.lora.request import LoRARequest
except ImportError:
    print("vLLM not installed")
    vllm = DummyImport()
    LoRARequest = lambda *args: args


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    path_model: str
    max_input_length: int = 512
    max_output_length: int = 512
    stopping_words: Optional[List[str]] = None

    def run(self, prompt: str) -> str:
        raise NotImplementedError


class VLLMModel(EvalModel):
    path_model: str
    path_lora: str = ""
    model: vllm.LLM = None
    quantization: Optional[str] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    tensor_parallel_size: int = 1

    def load(self):
        if self.model is None:
            self.model = vllm.LLM(
                model=self.path_model,
                trust_remote_code=True,
                quantization=self.quantization,
                enable_lora=self.path_lora != "",
                tensor_parallel_size=self.tensor_parallel_size,
            )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path_model)

    def format_prompt(self, prompt: str) -> str:
        self.load()
        prompt = prompt.rstrip(" ")  # Llama is sensitive (eg "Answer:" vs "Answer: ")
        return prompt

    def make_kwargs(self, do_sample: bool, **kwargs) -> dict:
        if self.stopping_words:
            kwargs.update(stop=self.stopping_words)
        params = vllm.SamplingParams(
            temperature=0.5 if do_sample else 0.0,
            max_tokens=self.max_output_length,
            **kwargs
        )

        outputs = dict(sampling_params=params, use_tqdm=False)
        if self.path_lora:
            outputs.update(lora_request=LoRARequest("lora", 1, self.path_lora))
        return outputs

    def run(self, prompt: str) -> str:
        prompt = self.format_prompt(prompt)
        outputs = self.model.generate([prompt], **self.make_kwargs(do_sample=False))
        pred = outputs[0].outputs[0].text
        pred = pred.split("<|endoftext|>")[0]
        return pred

    def run_batch(self, prompts: List[str]) -> List[str]:
        prompts = [self.format_prompt(p) for p in prompts]
        outputs = self.model.generate(prompts, **self.make_kwargs(do_sample=False))
        preds = []
        for o in outputs:
            preds.append(o.outputs[0].text.split("<|endoftext|>")[0])
        return preds

    def run_many(self, prompt: str, num: int) -> List[str]:
        prompt = self.format_prompt(prompt)
        outputs = self.model.generate(
            [prompt], **self.make_kwargs(do_sample=True, n=num)
        )
        preds = [o.text.split("<|endoftext|>")[0] for o in outputs[0].outputs]
        return preds


def test_model(prompt: str, path: str):
    model = VLLMModel(path_model=path)
    print(model.format_prompt(prompt))
    text = model.run(prompt)
    print(dict(prompt=prompt, text=text))


if __name__ == "__main__":
    Fire()
