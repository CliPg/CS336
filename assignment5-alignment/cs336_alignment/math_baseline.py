from vllm import LLM, SamplingParams
from typing import Callable, List

MATH_VALIDATION_PATH = "data/math/validation.jsonl"
examples = []
with open(MATH_VALIDATION_PATH, "r") as f:
    for line in f:
        examples.append(line)

prompt_path = "./prompts/r1_zero.prompt"
with open(prompt_path, "r") as f:
    prompt_template = f.read()

prompts = [prompt_template.format(question=example["question"]) for example in examples]

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str,str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)
    generated_texts = [output[0].text for output in outputs]
    for i, (example, generated_text) in enumerate(zip(examples, generated_texts)):
        reward = reward_fn(generated_text, example["answer"])
        print(f"Example {i+1}: Reward = {reward['reward']}, Generated = {generated_text}, Answer = {example['answer']}")
