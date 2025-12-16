from typing import Callable, List
import torch
from typing_extensions import Literal

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


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor
) -> torch.Tensor:
    """
    计算每个token的策略梯度损失

    Args:
        raw_rewards_or_advantages: 表示每个样本的奖励或已归一化的优势值,形状为 (batch_size, 1)
        policy_log_probs: 表示模型在生成每个token时的对数概率,形状为(batch_size, sequence_length)

    Return:
        loss: 形状为(batch_size, sequence_length)
    """
    # 将raw_rewards_or_advantages的形状广播成policy_log_probs
    raw_rewards_or_advantages = raw_rewards_or_advantages.expand_as(policy_log_probs)
    loss = -raw_rewards_or_advantages * policy_log_probs
    return loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float=0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    ratio = torch.exp(policy_log_probs - old_log_probs)
    """
    clip = policy_log_probs / old_log_probs
    if clip < 1 - cliprange:
        clip = 1 - cliprange
    elif clip > 1 + cliprange:
        clip = 1 + cliprange
    """
    clip_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    advantages =advantages.expand_as(policy_log_probs)
    unclipped_loss = ratio*advantages
    clipped_loss = clip_ratio*advantages
    loss = -torch.min(unclipped_loss, clipped_loss)

    clipped_mask = (clipped_loss < unclipped_loss).float()  # 被clip的为1，否则0
    metadata = {
        "clipped_ratio": clipped_mask.mean(),  # 比例（所有token中被clip的平均值）
        "clipped_mask": clipped_mask,           # 每个token是否被clip
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards must be provided for no_baseline loss"
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages must be provided for reinforce_with_baseline loss"
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages must be provided for grpo_clip loss"
        assert old_log_probs is not None, "old_log_probs must be provided for grpo_clip loss"
        assert cliprange is not None, "cliprange must be provided for grpo_clip loss"
        loss, metadata = compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange,
        )
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None
) -> torch.Tensor:
    
    masked_sum = (tensor * mask).sum(dim)
    masked_count = mask.sum(dim)
    masked_mean_tensor = masked_sum / masked_count

    return masked_mean_tensor


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    loss = masked_mean(tensor=loss, mask=response_mask)
    loss = loss.mean(0)
    loss = loss / gradient_accumulation_steps
    loss.backward()

    return loss, metadata