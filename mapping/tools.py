import torch
from torch import nn


def gradient_norm(model: nn.Module) -> float:
    norm = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            if m.weight is not None and m.weight.grad is not None:
                norm.append(torch.linalg.norm(m.weight.grad.detach()).item())
            if m.bias is not None and m.bias.grad is not None:
                norm.append(torch.linalg.norm(m.bias.grad.detach()).item())

    return sum(norm) / len(norm)
