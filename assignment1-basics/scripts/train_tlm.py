import os
import sys
import json
import torch
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cs336_basics.DL import data_loading
from cs336_basics.TLM import transformer_lm
from cs336_basics.AdamW import adamw_optimizer
from cs336_basics.CrossEntropy import cross_entropy
from cs336_basics.Checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.GC import gradient_clipping
from cs336_basics.LRS import learning_rate_schedule

# python train_tlm.py
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
TINY_STORIES_TRAIN = "TinyStoriesV2-GPT4-train.txt"
TINY_STORIES_VALID = "TinyStoriesV2-GPT4-valid.txt"

TRAIN_DATA_SAVE_PATH = os.path.join(DATA_DIR, "tiny_stories_train_tokens.dat")
VALID_DATA_SAVE_PATH = os.path.join(DATA_DIR, "tiny_stories_valid_tokens.dat")

CONFIG_PATH = "config.json"


def get_memmap_dataset(path, dtype=np.int32):
    arr = np.memmap(path, dtype=dtype, mode="r")
    return arr

def memmap_val_iterator(memmap_arr, batch_size, context_length):
    N = len(memmap_arr)
    nb = (N-context_length-1)//batch_size
    for bi in range(nb):
        base = bi*batch_size
        x = np.stack([memmap_arr[i:i+context_length] for i in range(base, base+batch_size)])
        y = np.stack([memmap_arr[i+1:i+context_length+1] for i in range(base, base+batch_size)])
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

import torch
import torch.nn as nn

def _to_device_and_compile(model: nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    return model, device

def main():
    # 1. 导入模型和配置
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    model = transformer_lm(**config['model'])

    params = {}
    for group in config.values():
        params.update(group)

    class DotDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = DotDict(params)
    model, device = _to_device_and_compile(model)

    os.makedirs(args.save_path, exist_ok=True)

    # 2. 加载数据集
    train_data = get_memmap_dataset(TRAIN_DATA_SAVE_PATH)
    val_data = get_memmap_dataset(VALID_DATA_SAVE_PATH)


    # 3. 构建优化器
    optimizer = adamw_optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. 恢复断点
    start_iter = 0
    if args.resume_checkpoint:
        resume_ckpt_path = pathlib.Path(__file__).resolve().parent.parent / f"checkpoints/ckpt_iter{args.resume_checkpoint}.pt"
        if resume_ckpt_path.exists():  # 检查文件是否存在
            print(f"Resuming from checkpoint {args.resume_checkpoint}")
            start_iter = load_checkpoint(resume_ckpt_path, model, optimizer)
            print(f"Resumed at iteration {start_iter}")
        else:
            print(f"Checkpoint {resume_ckpt_path} not found, starting from scratch.")


    # 5. 训练loop
    for iteration in tqdm(range(start_iter, args.train_steps), desc="Training"):
        model.train()
        x, y = data_loading(train_data, args.batch_size, args.context_len, device)
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.clip_grad_norm)
        
        # 更新学习率
        lr = learning_rate_schedule(
            iteration, args.lr, args.min_lr, args.warmup_iters, args.cosine_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # 验证
        if (iteration+1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                count = 0
                for x_val, y_val in memmap_val_iterator(val_data, args.batch_size, args.context_len):
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_logits = model(x_val)
                    val_loss = cross_entropy(
                        val_logits.reshape(-1, val_logits.shape[-1]),
                        y_val.reshape(-1)
                    )
                    val_losses.append(val_loss.item())
                    count += 1
                    if count >= args.val_batches:
                        break
                val_loss_mean = np.mean(val_losses)
                print(f"iter {iteration:05d}: VALID loss = {val_loss_mean:.4f}")

        # 保存
        if (iteration+1) % args.save_interval == 0:
            ckpt_name = os.path.join(args.save_path, f"ckpt_iter{iteration+1}.pt")
            save_checkpoint(model, optimizer, iteration+1, ckpt_name)
            print(f"Checkpoint saved to {ckpt_name}")

if __name__ == "__main__":
    main()