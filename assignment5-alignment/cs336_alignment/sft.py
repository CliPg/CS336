from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM
import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
import json
from sft_helper_methods import sft_microbatch_train_step, get_response_log_probs, tokenize_prompt_and_output


class SFTDataset(torch.utils.data.Dataset):
    def __init_(self, path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = [json.loads(line) for line in open(path, 'r')]

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        response = item['generated_text']
        
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


def supervised_finetuning(model_id: str, n_sft_steps: int, gradient_accumulation_steps: int):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    dataset = SFTDataset(path="data/sft_data.jsonl", tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    llm = init_vllm(model_id=model_id, device="cuda:1", seed=42)
    for step, batch in enumerate(dataloader):
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
        response_mask = tokenized_batch["response_mask"].to(model.device)
        # 计算log_probs
        policy_log_probs = get_response_log_probs(
            model=model,
            input_ids=input_ids,
            attention_mask=response_mask,
            return_entropy=False,
        )
        # 计算loss并进行优化
        loss, stats = sft_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            load_policy_into_vllm_instance(policy=model, llm=llm)

    
    
