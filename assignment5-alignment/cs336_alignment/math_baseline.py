from vllm import LLM, SamplingParams
from typing import Callable, List
import re
from tqdm import tqdm
import json
from drgrpo_grader import r1_zero_reward_fn

# CUDA_VISIBLE_DEVICES=1 python math_baseline.py

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str,str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    examples: List[dict],
) -> None:
    print("Starting evaluation...")
    outputs = vllm_model.generate(prompts, sampling_params=eval_sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]

    output_path = "eval_math_results_with_Qwen.jsonl"
    with open(output_path, "w") as f:
        for generated_text, example in tqdm(zip(generated_texts, examples), total=len(examples)):
            rewards = reward_fn(response=generated_text, ground_truth=example["solution"])
            result = {
                "problem": example["problem"],
                "solution": example["solution"],
                "generated_text": generated_text,
                "rewards": rewards
            }
            f.write(json.dumps(result) + "\n")
    print(f"Evaluation results saved to {output_path}")

def main():
    MATH_DATASET_PATH = "../data/competition_math_train.jsonl"

    # 加载数据集
    examples = []
    with open(MATH_DATASET_PATH, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    # 加载prompt，并将问题填入
    prompt_path = "./prompts/r1_zero.prompt"
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    prompts = [prompt_template.format(question=example["problem"]) for example in examples]

    model_name = "Qwen/Qwen2.5-Math-1.5B"
    vllm_model = LLM(model=model_name, tensor_parallel_size=1)

    eval_sampling_params = SamplingParams(temperature=0.1, max_tokens=1024, top_p=1.0, stop=["</answer>"])

    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=eval_sampling_params,
        examples=examples,
    )

if __name__ == "__main__":
    main()
    