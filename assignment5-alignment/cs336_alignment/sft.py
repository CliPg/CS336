from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM
import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
import json
from sft_helper_methods import sft_microbatch_train_step, get_response_log_probs, tokenize_prompt_and_output
import wandb
from tqdm import tqdm
import os

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = [json.loads(line) for line in open(path, 'r')]

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item['prompt'],
            "generated_text": item['generated_text']
        }
        
    def __len__(self):
        return len(self.data)

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    启动 vLLM 推理进程（将其放到单独 GPU 上）
    """
    vllm_set_random_seed(seed)

    # 从 TRL（HuggingFace）Monkeypatch
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    从 TRL 仓库复制的辅助函数：
    将 训练中的policy 模型权重加载到 vLLM 推理实例中
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def save_checkpoint(step, model, optimizer, checkpoint_dir="../models/sft/checkpoints"):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at step {step}: {checkpoint_path}")


def load_latest_checkpoint(model, optimizer, checkpoint_dir="../models/sft/checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return 0  # 无checkpoint
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        return 0
    latest_ckpt = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
    path = os.path.join(checkpoint_dir, latest_ckpt)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Resumed from checkpoint: {path}")
    return ckpt["step"]

def supervised_finetuning(model_id: str, n_sft_steps: int, gradient_accumulation_steps: int):
    wandb.define_metric("train_step")
    wandb.define_metric("train/*", step_metric="train_step")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    start_step = load_latest_checkpoint(model, optimizer)
    dataset = SFTDataset(path="../data/eval_math_results_with_Qwen.jsonl", tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    llm = init_vllm(model_id=model_id, device="cuda:0", seed=42)
    for step, batch in enumerate(tqdm(dataloader, desc="Training", total=len(dataloader))):
        step += start_step
        if step >= n_sft_steps:
            break
        # 加载数据
        prompt_strs = batch["prompt"]
        output_strs = batch["generated_text"]
        # 对prompt和output进行tokenize
        tokenized_batch = tokenize_prompt_and_output(
            prompt_strs=prompt_strs,
            output_strs=output_strs,
            tokenizer=tokenizer,
        )
        input_ids = tokenized_batch["input_ids"].to(model.device)
        labels = tokenized_batch["labels"].to(model.device)
        response_mask = tokenized_batch["response_mask"].to(model.device)
        # 计算log_probs
        policy_log_probs = get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
            return_entropy=False,
        )["log_probs"]
        # 计算loss并进行优化
        loss, stats = sft_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({
                "train/loss_sum": stats["loss_sum"].item(),
                "train/n_tokens": stats["n_tokens"].item(),
                "train/avg_ce_per_token": stats["avg_ce_per_token"].item(),
                "train/mean_log_prob": stats["mean_log_prob"].item(),
                "train_step": step + 1
            })
        if (step + 1) % 100 == 0:
            load_policy_into_vllm_instance(policy=model, llm=llm)
            save_checkpoint(step + 1, model, optimizer)
    model.save_pretrained("../models/sft/sft_finetuned_model")
    tokenizer.save_pretrained("../models/sft/sft_finetuned_tokenizer")

    
if __name__ == "__main__":
    wandb.init(project="cs336-sft", name="sft-experiment-1")
    supervised_finetuning(
        model_id="Qwen/Qwen2.5-Math-1.5B",
        n_sft_steps=1000,
        gradient_accumulation_steps=4,
    )
    wandb.finish()
