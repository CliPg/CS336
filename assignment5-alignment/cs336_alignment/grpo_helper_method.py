from typing import Callable, List
import torch

def compute_group_normalized_rewards(
    reward_fn : Callable[[str, str], dict[str, float]],
    rollout_responses : List[str],
    repeated_ground_truths : List[str],
    group_size : int,
    advantage_eps : float,
    normalize_by_std : bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    计算每组 rollout 响应的奖励值,并根据组大小(group size)进行归一化处理。

    Args:
        reward_fn: 奖励函数
        rollout_responses: 策略模型生成的rollout响应列表, 列表的长度为
                           rollout_batch_size = n_prompts_per_rollout_batch * group_size,
                           n_prompts_per_rollout_batch表示示例的数量
        repeated_ground_truths: 每个示例对应的真实答案。列表长度与rollout_responses一样。因为每个真实答案会重复group_size次。
        group_size: 每个问题对应的响应数量。

    Return:
        advantages: 形状为rollout_batch_size的tensor
        raw_rewards: 形状为rollout_batch_size的tensor
        metadata: 字典,奖励的均值,标准差
    """
    raw_rewards = []
    for rollout_response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward = reward_fn(rollout_response, ground_truth)
        raw_rewards.append(reward["reward"])
    raw_rewards_tensor = torch.tensor(raw_rewards)
    raw_rewards_tensor = raw_rewards_tensor.view(-1, group_size)

    group_mean = raw_rewards_tensor.mean(dim=1, keepdim=True)
    group_std = raw_rewards_tensor.std(dim=1, keepdim=True)

    if normalize_by_std:
        advantages_tensor = (raw_rewards_tensor - group_mean) / (group_std + advantage_eps)
    else:
        advantages_tensor = raw_rewards_tensor - group_mean

    advantages_tensor = advantages_tensor.view(-1)
    raw_rewards_tensor = raw_rewards_tensor.view(-1)

    metadata = {
        "reward_mean": raw_rewards_tensor.mean().item(),
        "reward_std": raw_rewards_tensor.std().item(),
        "reward_max": raw_rewards_tensor.max().item(),
        "reward_min": raw_rewards_tensor.min().item(),
    }

    return advantages_tensor, raw_rewards_tensor, metadata
