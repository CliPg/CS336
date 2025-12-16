from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import torch
import timeit


def benchmarking_script(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    batch_size: int = 32,
    warmup_steps: int = 5,
    benchmark_steps: int = 20,
    backward: bool = True,
    device: str = "cuda",
):
    """
    profiling the model by timing the forward and backward passes

    Args:
        d_model (int): dimension of model
        d_ff (int): dimension of feedforward layer
        num_layers (int): number of layers in the model
        num_heads (int): number of attention heads        
    """
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)

    optimizer = AdamW(model.parameters())

    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    print("Warming up...")
    for _ in range(warmup_steps):
        logits = model(x)

        if backward:
            loss = cross_entropy(logits, y)
            loss.backward()
            optimizer.zero_grad()
        
        torch.cuda.synchronize() # 让 CPU 等待 GPU 把当前所有任务全部执行完

    print("Starting benchmarking...")

    start = timeit.default_timer()

    for _ in range(benchmark_steps):
        logits = model(x)

        if backward:
            loss = cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        torch.cuda.synchronize()

    end = timeit.default_timer()

    total_time = end - start
    avg_time = total_time / benchmark_steps


    print(f"\nBenchmark completed:")
    print(f"  Steps: {benchmark_steps}")
    print(f"  Mode: {'Forward+Backward' if backward else 'Forward only'}")
    print(f"  Total time: {total_time:.4f} s")
    print(f"  Avg time per step: {avg_time:.4f} s")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    benchmarking_script(
        vocab_size=32000,
        context_length=1024,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        rope_theta=10000.0,
        batch_size=32,
        warmup_steps=5,
        benchmark_steps=20,
        backward=True,
        device="cuda",
    )