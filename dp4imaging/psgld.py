"""Implementation of preconditioned stochastic gradient Langevin dynamics.

Typical usage example:

sampler = PreconditionedSGLD(g.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)
sampler.step()
"""

import torch
from torch.optim.optimizer import Optimizer
import typing

class PreconditionedSGLD(Optimizer):
    """Preconditioned stochastic gradient Langevin dynamics.

    Implements the Preconditioned SGLD MCMC sampling algorithm according to
    https://arxiv.org/abs/1512.07666. The implementation is based on the PyTorch
    v1.3 RMSprop implementation
    (https://github.com/pytorch/pytorch/blob/v1.3.0/torch/optim/rmsprop.py)

    Attributes:
        param_groups: List of dictionaries containing the parameters. state:
        Dictionary containing the state of the optimizer.
    """

    def __init__(self,
                 params: typing.Generator,
                 lr: float = 1e-2,
                 beta: float = 0.99,
                 Lambda: float = 1e-15,
                 weight_decay: float = 0.0):
        """Initializes a PreconditionedSGLD object.

        Args:
            params: List of dictionaries containing the parameters.
            lr: A float representing the learning rate.
            beta: A float representing the momentum parameter.
            Lambda: A float to avoid division by zero.
            weight_decay: A float representing the weight decay parameter,
                i.e., Gaussian prior precision.
        """
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if Lambda < 0.0:
            raise ValueError(f'Invalid Lambda value: {Lambda}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if beta < 0.0:
            raise ValueError(f'Invalid beta value: {beta}')

        defaults = dict(lr=lr,
                        beta=beta,
                        Lambda=Lambda,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        """Performs a single sampling step.

        See Algorithm 1 in https://arxiv.org/abs/2110.04825.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization.
                if len(state) == 0:
                    state['step'] = 0
                    state['v'] = torch.zeros_like(p.data)

                v = state['v']
                beta = group['beta']
                state['step'] += 1

                # Including the gradient contribution of the Gaussian prior.
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Computing the preconditioner.
                v.mul_(beta).addcmul_(grad, grad, value=1 - beta)
                m_inv = v.sqrt().add_(group['Lambda'])

                # Taking a preconditioned gradient step.
                p.data.addcdiv_(grad, m_inv, value=-group['lr'])

                # Adding noise with variance `2 * group['lr'] / m_inv` to
                # `params`.
                noise_std = 2 * group['lr'] / m_inv
                noise_std = noise_std.sqrt()
                noise = p.data.new(p.data.size()).normal_(mean=0, std=1)
                noise *= noise_std
                p.data.add_(noise)
