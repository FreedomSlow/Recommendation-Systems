import torchmetrics.functional

import utils
import torch


def eval_model(net, data_iter, metric_func, device=None):
    """Simple metric evaluator for neural networks"""
    if device is None:
        device = utils.try_gpu()

    if isinstance(net, torch.nn.Module):
        net.eval()

    metrics = utils.SaveMetrics(2)

    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)

            metrics.add(metric_func(y, net(X)), y.numel())

    return metrics[0] / metrics[1]


