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
    # # pt表示把分词后的结果转换成 PyTorch Tensor格式
    # # padding=True 表示自动把所有句子补齐
    # # truncation=True 表示当句子太长超过模型最大输入长度时，自动截断
    # # max_length 表示最大输入长度
    # tokenized_prompts = tokenizer(prompt_strs, add_special_tokens=False)
    # tokenized_outputs = tokenizer(output_strs, add_special_tokens=False)
    # """
    # tokenizer的输出结构
    # {
    #     'input_ids': [151643, 30946, 16617, 29991],     # 每个token对应词表中的整数ID
    #     'token_type_ids': [0, 0, 0, 0],                 # 可选字段（部分模型用来区分句子）
    #     'attention_mask': [1, 1, 1, 1]                  # padding部分为0,有效token为1
    # }
    # """
    # input_ids = []
    # # labels = []
    # response_mask = []
    # for prompt_ids, output_ids in zip(tokenized_prompts['input_ids'], tokenized_outputs['input_ids']):
    #     # 去掉output_ids中的padding部分
    #     # padding只是为了对齐，不去掉loss会出错
    #     """
    #     因果语言模型(causal LM)训练的思路,input需要包含问题和答案
    #     假设我们有一个问题 Q 和答案 A:
	# 	我们希望模型学习：看到问题 Q 后，生成答案 A。
	# 	如果只把 Q 作为 input,而 labels 只包含 A,模型就不知道 预测 A 时应该基于 Q 的上下文。
	# 	因此训练时通常把 Q + A 拼接：
    #     """
    #     # 去掉 prompt 和 output 的 padding
    #     # prompt_ids = prompt_ids[prompt_ids != tokenizer.pad_token_id]
    #     # output_ids = output_ids[output_ids != tokenizer.pad_token_id]
    #     
    #     # 拼接prompt和output的input_ids
    #     combined_ids = prompt_ids + output_ids
    #     input_ids.append(torch.tensor(combined_ids, dtype=torch.long))
    #     
    #     # 构建labels，prompt部分为-100，output部分为对应的token id
    #     # prompt部分是模型的输入，但我们不希望模型去预测prompt本身
    #     # prompt_labels = torch.full_like(prompt_ids, -100)  # 创建一个和 prompt_ids 形状相同的 tensor，每个元素都是 -100。-100表示忽略这个位置的loss。
    #     # combined_labels = torch.cat([prompt_labels, output_ids])
    #     # labels.append(combined_labels)
    #     
    #     # 构建response_mask，prompt部分为0，output部分为1
    #     response_mask = [0] * len(prompt_ids) + [1] * len(output_ids)
    #     response_mask.append(torch.tensor(response_mask, dtype=torch.long))
# 
    # batch_size = len(input_ids)
    # max_len = max(len(ids) for ids in input_ids)
    # input_ids_batch = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
    # response_mask_batch = torch.zeros((batch_size, max_len), dtype=torch.long)
# 
    # for i, (ids, mask) in enumerate(zip(input_ids, response_mask)):
    #     input_ids_batch[i, :len(ids)] = ids
    #     response_mask_batch[i, :len(ids)] = mask
    # # 使用pad_sequence对input_ids, labels, response_mask进行padding
    # # batch_first=True表示输出的tensor的第一个维度是batch_size
    # # input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    # # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    # # response_mask = torch.nn.utils.rnn.pad_sequence(response_mask, batch_first=True, padding_value=0)
# 
    # return {
    #     "input_ids": input_ids_batch[:, :-1],
    #     "labels": input_ids_batch[:, 1:],
    #     "response_mask": response_mask_batch[:, :-1]
    # }
    input_ids_list = []
    response_mask_list = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_enc = tokenizer(prompt, add_special_tokens=False)
        output_enc = tokenizer(output, add_special_tokens=False)
        full_input = prompt_enc['input_ids'] + output_enc['input_ids']
        response_mask = [0] * len(prompt_enc['input_ids']) + [1] * len(output_enc['input_ids'])
        input_ids_list.append(torch.tensor(full_input, dtype=torch.long))
        response_mask_list.append(torch.tensor(response_mask, dtype=torch.long))

    batch_size = len(input_ids_list)
    max_len = max(len(ids) for ids in input_ids_list)
    input_ids_batch = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
    response_mask_batch = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids_list, response_mask_list)):
        seq_len = len(ids)
        input_ids_batch[i, :seq_len] = ids
        response_mask_batch[i, :seq_len] = mask

    return {
        "input_ids": input_ids_batch[:, :-1],               # (batch, max_len-1)
        "labels": input_ids_batch[:, 1:],                   # (batch, max_len-1)
        "response_mask": response_mask_batch[:, 1:]         # (batch, max_len-1)
    }

    

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logz = torch.logsumexp(logits, dim=-1, keepdim=True)
    p = torch.nn.functional.softmax(logits, dim=-1)
    expected_logits = (p * logits).sum(dim=-1, keepdim=True)
    entropy = logz - expected_logits
    return entropy.squeeze(-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    # 计算每个token的对数概率
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # 选出每个位置对应的token的对数概率
    log_probs_for_labels = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result = {
        "log_probs": log_probs_for_labels,
    }
    if return_token_entropy:
        entropy = compute_entropy(logits)
        result["token_entropy"] = entropy
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    # 用乘法运算实现mask
    masked_tensor = tensor * mask
    sum_masked_tensor = masked_tensor.sum(dim=dim)
    normalized_tensor = sum_masked_tensor / normalize_constant
    return normalized_tensor
    

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, seq_len = policy_log_probs.shape
    ce_loss = -policy_log_probs
    loss_sum = masked_normalize(
        tensor=ce_loss,
        mask=response_mask,
        normalize_constant=normalize_constant,
    )
    loss = loss_sum / gradient_accumulation_steps / batch_size
    loss.backward()

    n_tokens = response_mask.sum()
    avg_token_ce = loss_sum / (n_tokens + 1e-8)
    stats = {
        "loss_sum": loss_sum.detach(),
        "n_tokens": n_tokens.detach(),
        "avg_ce_per_token": avg_token_ce.detach(),
        "mean_log_prob": (policy_log_probs * response_mask).sum() / (n_tokens + 1e-8)
    }

    return loss.detach(), stats

    