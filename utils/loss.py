import torch
import torch.nn as nn
from utils.activations import log_softmax
import time

from torch import Tensor

def CrossEntropyLoss(outputs: Tensor, targets: Tensor) -> Tensor:
    if targets.shape[1] != outputs.shape[1]:
        raise ValueError("Targets must be one-hot encoded with the same number of columns as classes in outputs")
    log_softmax_outputs = torch.nn.functional.log_softmax(outputs, dim=1)    
    loss = -torch.sum(log_softmax_outputs * targets) / outputs.shape[0]
    
    return loss



if __name__ == ("__main__"):
 
    outputs = torch.randn(1000, 10, requires_grad=True)  # 1000 ví dụ, 10 lớp
    targets = torch.randint(0, 10, (1000,))

    start_time = time.time()
    loss_custom = CrossEntropyLoss(outputs, targets)
    end_time = time.time()
    time_custom = end_time - start_time
    print(f"Custom CrossEntropyLoss: {loss_custom.item()} (Time: {time_custom:.6f} seconds)")

    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    loss_pytorch = criterion(outputs, targets)
    end_time = time.time()
    time_pytorch = end_time - start_time
    print(f"PyTorch CrossEntropyLoss: {loss_pytorch.item()} (Time: {time_pytorch:.6f} seconds)")

    # So sánh thời gian
    print(f"Time difference: {time_custom - time_pytorch:.6f} seconds")

