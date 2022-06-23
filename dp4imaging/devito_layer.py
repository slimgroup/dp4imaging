import torch
import numpy as np


class ForwardBorn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, src, model, geometry, solver, device):

        ctx.model = model
        ctx.geometry = geometry
        ctx.solver = solver
        ctx.device = device
        ctx.src = src

        # Prepare input
        input = input.cpu().detach()
        input = torch.nn.ReplicationPad2d((ctx.model.nbl))(input).numpy()

        # Linearized forward modeling
        d_lin = ctx.solver.born(input[0, 0, :, :], src=src)[0].data

        return torch.from_numpy(np.array(d_lin)).to(ctx.device)

    @staticmethod
    def backward(ctx, grad_output):

        grad_output = grad_output.detach().cpu().numpy()
        rec = ctx.geometry.rec
        rec.data[:] = grad_output[:]

        # Adjoint linearized modeling
        u0 = ctx.solver.forward(src=ctx.src, save=True)[1]
        g = ctx.solver.gradient(rec, u0)[0].data

        # Remove padding
        nb = ctx.model.nbl
        g = torch.from_numpy(np.array(g[nb:-nb, nb:-nb])).to(ctx.device)
        g = g.view(1, 1, g.shape[0], g.shape[1])

        return g, None, None, None, None, None
