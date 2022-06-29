"""Devito's linearized Born scattering forward operator as a Pytorch "layer".

This implementation is based on the original implementation of Devito4PyTorch at
https://github.com/slimgroup/Devito4PyTorch.

Typical usage example:

forward_born = ForwardBorn()
y_pred = forward_born.apply(x, src, model, geometry, solver, device)

# Some objective function.
obj = objective_function(y_obs, y_pred)

# The following computes the gradient of the objictive function with respect to
# the seismic image `x`, which uses `ForwardBorn.backward`` method.
obj.backward ()
"""

from examples.seismic import Model, AcquisitionGeometry, WaveletSource
from examples.seismic.acoustic import AcousticWaveSolver
import numpy as np
import torch
from typing import Tuple


class ForwardBorn(torch.autograd.Function):
    """PyTorch wrapper for Devito's linearized Born scattering forward operator.

    This module allows to have access to the automatic differentiation utilities
    of PyTorch while exploiting Devito’s highly optimized migration and
    demigration operators.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, src: WaveletSource, model: Model,
                geometry: AcquisitionGeometry, solver: AcousticWaveSolver,
                device: torch.device) -> torch.Tensor:
        """Linearized Born scattering forward operator.

        Args:
            x: A torch.Tensor containing the input seismic image.
            src: A WaveletSource object containing the source signature.
            model: A Model object containing the finite-difference grid and
                wave-equation coefficients.
            geometry: An AcquisitionGeometry object containing the receiver
                and source positions.
            solver: An AcousticWaveSolver object that contains the
                highly-optimized finite-difference solver.
            device: A torch.device object that specifies the device that
                seismic image is on.

        Returns:
            A torch.Tensor containing the output linearized seismic data.
        """
        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device
        ctx.src = src

        # Move seismic image to cpu, cast to numpy array, and include
        # finite-difference related boundary padding.
        x = x.cpu().detach()
        x = torch.nn.ReplicationPad2d((ctx.model.nbl))(x).numpy()

        # Apply Devito's linearized forward modeling operator to the image.
        d_lin = ctx.solver.born(x[0, 0, :, :], src=src)[0].data

        return torch.from_numpy(np.array(d_lin)).to(ctx.device)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[torch.Tensor]:
        """Adjoint of the linearized Born scattering forward operator.

        Args:
           dy: A torch.Tensor containing the data residual.

        Returns:
            A torch.Tensor containing gradient with respect to the input
                seismic image.
        """
        # Move data residual image to cpu, cast to numpy array.
        dy = dy.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = dy[:]

        # Apply the adjoint of the linearized modeling operator.
        u0 = ctx.solver.forward(src=ctx.src, save=True)[1]
        dx = ctx.solver.gradient(rec, u0)[0].data

        # Undo the padding and cast to torch.Tensor.
        nb = ctx.model.nbl
        dx = torch.from_numpy(np.array(dx[nb:-nb, nb:-nb])).to(ctx.device)
        dx = dx.view(1, 1, dx.shape[0], dx.shape[1])

        return dx, None, None, None, None, None
