from typing import Callable

def compute_group_normalized_rewards(
    reward_fn : Callable[[str, str], dict[str, float]],
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    pass