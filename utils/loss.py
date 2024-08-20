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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure inputs and targets are on the same device
        device = inputs.device

        # Apply softmax to the model outputs
        inputs = torch.softmax(inputs, dim=1)

        # Compute the focal loss components
        targets = torch.argmax(targets, dim=1)
        batch_size = targets.size(0)
        num_classes = inputs.size(1)

        # One-hot encode the targets and move to the same device as inputs
        targets_one_hot = torch.eye(num_classes, device=device)[targets]

        # Compute the focal loss
        alpha = torch.tensor(self.alpha, device=device)
        gamma = torch.tensor(self.gamma, device=device)

        # Compute cross entropy loss
        cross_entropy_loss = -targets_one_hot * torch.log(inputs + 1e-8)  # Added epsilon to avoid log(0)
        
        # Compute the focal loss
        focal_loss = alpha * torch.pow(1 - inputs, gamma) * cross_entropy_loss
        
        if self.reduction == 'mean':
            return torch.sum(focal_loss) / batch_size
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

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

