import torch
from torch import Tensor


def softmax(x: Tensor) -> Tensor:
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)
    return exp_x / sum_x

def log_softmax(x: Tensor) -> Tensor:
    return x - torch.logsumexp(x, dim=1, keepdim=True)
