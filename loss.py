import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty

def r1_penalty(real_pred, real_img):
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.size(0), -1).sum(1).mean()
    return grad_penalty
