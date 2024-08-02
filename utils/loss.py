import torch
import torch.nn as nn
from activations import log_softmax
import time

from torch import Tensor

def CrossEntropyLoss(outputs: Tensor, targets: Tensor) -> Tensor:
    outputs = log_softmax(outputs)
    outputs = outputs[range(targets.shape[0]), targets]
    return -torch.sum(outputs) / targets.shape[0]


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

