import torch
import numpy.typing as npt
import numpy as np
def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,)
    )
    
    inputs = np.stack([dataset[start:start+context_length] for start in starts])
    labels = np.stack([dataset[start+1:start+context_length+1] for start in starts])
    
    return (
        torch.from_numpy(inputs).long().to(device),
        torch.from_numpy(labels).long().to(device)
    )