import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    对 prompt 和 output 字符串进行分词，并构建 response_mask。
    
    Args:
        prompt_strs: list[str]   # prompt 字符串列表
        output_strs: list[str]   # output 字符串列表
        tokenizer: PreTrainedTokenizer  # 用于分词的 tokenizer
    
    Returns:
        dict[str, torch.Tensor]  # 包含 input_ids, labels, response_mask
    """
    # pt表示把分词后的结果转换成 PyTorch Tensor格式
    # padding=True 表示自动把所有句子补齐
    # truncation=True 表示当句子太长超过模型最大输入长度时，自动截断
    # max_length 表示最大输入长度
    tokenized_prompts = tokenizer(prompt_strs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    tokenized_outputs = tokenizer(output_strs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    """
    tokenizer的输出结构
    {
        'input_ids': [151643, 30946, 16617, 29991],     # 每个token对应词表中的整数ID
        'token_type_ids': [0, 0, 0, 0],                 # 可选字段（部分模型用来区分句子）
        'attention_mask': [1, 1, 1, 1]                  # padding部分为0,有效token为1
    }
    """
    input_ids = []
    labels = []
    response_mask = []
    for prompt_ids, output_ids in zip(tokenized_prompts['input_ids'], tokenized_outputs['input_ids']):
        # 去掉output_ids中的padding部分
        # padding只是为了对齐，不去掉loss会出错
        """
        因果语言模型(causal LM)训练的思路,input需要包含问题和答案
        假设我们有一个问题 Q 和答案 A:
		我们希望模型学习：看到问题 Q 后，生成答案 A。
		如果只把 Q 作为 input,而 labels 只包含 A,模型就不知道 预测 A 时应该基于 Q 的上下文。
		因此训练时通常把 Q + A 拼接：
        """
        output_ids = output_ids[output_ids != tokenizer.pad_token_id]
        
        # 拼接prompt和output的input_ids
        combined_ids = torch.cat([prompt_ids, output_ids])
        input_ids.append(combined_ids)
        
        # 构建labels，prompt部分为-100，output部分为对应的token id
        # prompt部分是模型的输入，但我们不希望模型去预测prompt本身
        prompt_labels = torch.full_like(prompt_ids, -100)  # 创建一个和 prompt_ids 形状相同的 tensor，每个元素都是 -100。-100表示忽略这个位置的loss。
        combined_labels = torch.cat([prompt_labels, output_ids])
        labels.append(combined_labels)
        
        # 构建response_mask，prompt部分为0，output部分为1
        prompt_mask = torch.zeros_like(prompt_ids)  # prompt部分为0
        output_mask = torch.ones_like(output_ids)   # output部分为1
        combined_mask = torch.cat([prompt_mask, output_mask])
        response_mask.append(combined_mask)

    # 使用pad_sequence对input_ids, labels, response_mask进行padding
    # batch_first=True表示输出的tensor的第一个维度是batch_size
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    response_mask = torch.nn.utils.rnn.pad_sequence(response_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }
    

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logz = torch.logsumexp(logits, dim=-1, keepdim=True)
    p = torch.nn.functional.softmax(logits, dim=-1)
    expected_logits = (p * logits).sum(dim=-1, keepdim=True)
    entropy = logz - expected_logits
    return entropy.squeeze(-1)


def get_response_log_probs(model, input_ids, attention_mask=None, return_entropy=False):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    # 计算每个token的对数概率
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # 选出每个位置对应的token的对数概率
    token_log_probs = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    if return_entropy:
        entropy = compute_entropy(logits)
        return token_log_probs, entropy
    return token_log_probs


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    # 用乘法运算实现mask
    masked_tensor = tensor * mask
    sum_masked_tensor = masked_tensor.sum(dim=dim, keepdim=True)
    normalized_tensor = sum_masked_tensor / normalize_constant
    return normalized_tensor
    

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    normalized_log_probs = masked_normalize(
        policy_log_probs,
        response_mask,
        normalize_constant,
        dim=1,
    )
    loss = -normalized_log_probs.sum() / gradient_accumulation_steps
    stats = {
        "sft_loss": loss.detach(),
        "sft_mean_log_prob": (policy_log_probs * response_mask).sum() / response_mask.sum(),
    }
    return loss, stats

