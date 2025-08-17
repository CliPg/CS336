import torch
import os
from typing import IO, Any, BinaryIO

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    if isinstance(out, (str, os.PathLike)):
        with open(out, 'wb') as f:
            torch.save(checkpoint, f)
    else:
        torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    if isinstance(src, (str, os.PathLike)):
        with open(src, 'rb') as f:
            checkpoint = torch.load(f)
    else:
        checkpoint = torch.load(src)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']