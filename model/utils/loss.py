import torch
import torch.nn.functional as F

from model.utils import metric_utils
import numpy as np

def inf(*args):
    return torch.as_tensor(float("Inf"))


def gradient_loss(s):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :]) ** 2
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1]) ** 2
    return (torch.mean(dx) + torch.mean(dy)) / 2.0


def multitask_loss(warp, flow, output, input_fixed, target_fixed):
    lung_mask = torch.zeros_like(target_fixed)
    lung_mask[target_fixed != 0] = 1
    warp = warp * lung_mask
    input_fixed = input_fixed * lung_mask
    recon_loss = mse(warp, input_fixed)
    grad_loss = gradient_loss(flow)
    seg_loss = mse(output, target_fixed)

    return recon_loss + 0.01 * grad_loss + seg_loss


def deformation_loss(warp, flow, input_fixed):
    recon_loss = mse(warp, input_fixed)
    grad_loss = gradient_loss(flow)
    return recon_loss + 0.01 * grad_loss


def l1(output, target):
    return F.l1_loss(output, target)


def mse(output, target):
    return F.mse_loss(output, target)


def mse_difference(output, target, output_ref, target_ref, outDiff, groundDiff):

    return F.mse_loss(output, target) + F.mse_loss(output_ref, target_ref) + F.mse_loss(outDiff, groundDiff)


def nll_loss(output, target):
    return F.nll_loss(metric_utils.flatten(output), metric_utils.flatten(target))


def dice_loss(output, target, weights):
    size = output.size()
    outputs = torch.zeros_like(output)
    targets = torch.zeros_like(target)
    for i in range(size[0]):
        for j in range(size[1]):
            outputs[i][j] = output[i][j] * weights[j]
            targets[i][j] = target[i][j] * weights[j]
    return metric_utils.asymmetric_loss(1, output, target)


def asymmetric_loss(output, target):
    return metric_utils.asymmetric_loss(2, output, target)
