import torch
import torch.nn.functional as F
import numpy as np


def bprl(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalized Ranking Loss
    https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf
    """

    dist = positive - negative
    return -F.logsigmoid(dist).mean()


def hinge_loss_rec(positive: torch.Tensor, negative: torch.Tensor, margin=1) -> torch.Tensor:
    """Hinge loss for recommendations"""
    dist = positive - negative
    return torch.sum(torch.maximum(margin - dist, torch.Tensor([0])))
