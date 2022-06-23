"""Implementation of deep Bayesian inference for seismic imaging algorithm.

Typical usage example:

imaging_instance = DeepPriorImaging(args)
imaging_instance.sample(args)
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm

from dp4imaging.deep_prior import DeepPrior
from dp4imaging.psgld import PreconditionedSGLD
from dp4imaging.setup_experiment import SeismicSetup
from dp4imaging.utils import setup_sample_file, save_checkpoint, decay_fn


class DeepPriorImaging(object):
    """Seismic imaging and uncertainty quantification with deep priors.

    This class implements the seismic imaging and uncertainty quantification
    approach with deep priors. Sets up the seismic experiment, creates the
    observed data and forward operators, and initializes the deep prior. During
    training, the deep prior is updated to fit the observed data. The code saves
    all the seismic images throughout posterior sampling.

    Attributes:
        device: A torch.device object indicating the device to use.
        imaging_setup: A SeismicSetup object, containing the seismic experiment
            acquisition setup.
        obj_log: A list of objective function values.
        error_log: A list of prediction error values.
    """

    def __init__(self, args: argparse.Namespace):
        """Initializes a DeepPriorImaging object.

        Args:
            args: An argparse.Namespace, containing command line arguments.
        """
        super().__init__()

        # Setting device and default data type.
        if torch.cuda.is_available() and args.cuda:
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Setup seismic experiment, including data acquisition setup and forward
        # operators.
        self.imaging_setup = SeismicSetup(self.device,
                                          args.sigma,
                                          sim_source=True)

        # Set up HDF5 file to store posterior samples.
        setup_sample_file(args, self.imaging_setup.dm.shape[2:], args.max_itr)

        # Initialize some book keeping lists.
        self.obj_log = []
        self.error_log = []

    def nll(self, d_pred: torch.Tensor, d_obs: torch.Tensor,
            sigma: float) -> torch.Tensor:
        """Negative-log likelihood under Gaussian noise assumption.

        Args:
            d_pred: A torch.Tensor object, containing the predicted data. d_obs:
            A torch.Tensor object, containing the observed data. sigma: A float,
            containing the noise standard deviation.

        Returns:
            A torch.Tensor containing a float value, the negative log
                likelihood.
        """
        return (self.imaging_setup.nsrc / 2.0) * torch.norm(
            (d_pred - d_obs) / sigma)**2

    def sample(self, args: argparse.Namespace):
        """Main deep-prior based seismic imaging posterior sampling loop.

        Args:
            args: An argparse.Namespace, containing command line hyperparameters
                and data paths.
        """
        # Create observed data (HDF5 dataset, not fully loaded into memory).
        d_obs = self.imaging_setup.create_sim_src_data()

        # True seismic image (perturbation model) to be used for book keeping.
        dm = (self.imaging_setup.dm).to(self.device)

        # Deep prior network and its fixed random input.
        g = DeepPrior(dm.size()).to(self.device)
        z = torch.randn(g.get_latent_shape()).to(self.device)

        # Stochastic gradient Langevin dynamics sampling sub-routine.
        self.sampler = PreconditionedSGLD(g.parameters(),
                                          args.lr,
                                          weight_decay=args.wd)

        # Learning rate decay function.
        self.lr_fn = decay_fn(args)

        # A buffer for samples to be written to file every 100 iterations.
        samples_buffer = []

        # Sampling loop, run for `args.max_itr` iterations.
        with tqdm(range(args.max_itr), unit='iteration',
                  colour='#B5F2A9') as pb:
            for itr in pb:

                # Update the learning rate.
                self.lr_decay(itr)

                # Randomly pick a source experiment
                idx = np.random.choice(self.imaging_setup.nsrc,
                                       1,
                                       replace=False)[0]

                # Create Devito-based Born scattering forward modeling operator
                # for source position index `idx`. This operator is wrapped as a
                # pytorch layer to facilitate gradient computation via automatic
                # differentiation while exploiting Devito's highly optimized
                # stencil code.
                forward_op = self.imaging_setup.create_op(src_idx=idx)

                # Compute predicted seismic image via the deep prior
                # reparameterization.
                dm_est = g(z)

                # Compute predicted data to match the observed data.
                d_pred = forward_op(dm_est)

                # Compute the negative-log likelihood.
                obj = self.nll(d_pred,
                               torch.from_numpy(d_obs[idx]).to(self.device),
                               args.sigma)

                # Compute the gradient of negative-log likelihood with respect
                # to deep prior weights. The gradient contribution of the
                # Gaussian prior term on deep prior weights will be included
                # internally in the sampling sub-routine.
                grad = torch.autograd.grad(obj, g.parameters())
                for param, grad in zip(g.parameters(), grad):
                    param.grad = grad

                # Update the deep prior weights. The gradient of the Gaussian
                # prior is included internally in `sampler`.
                self.sampler.step()

                # Write summary Of this iterations.
                self.write_summary(args, itr, obj.item(), dm_est, dm, pb)

                samples_buffer.append(dm_est.detach().cpu().numpy())

                if itr % 100 == 0 or itr == args.max_itr - 1:
                    save_checkpoint(args, self.obj_log, self.error_log, g, z,
                                    samples_buffer)
                    samples_buffer = []

    def write_summary(self, args: argparse.Namespace, itr: int, obj: float,
                      dm_est: np.ndarray, dm: np.ndarray, pb: tqdm):
        """Writes the summary of one SGLD iteration to file.

        Args:
            args: An argparse.Namespace, containing command line hyperparameters
                and data paths.
            itr: An int, containing the current iteration number.
            obj: A float, containing the negative-log likelihood.
            dm_est: A numpy.ndarray, containing the estimated seismic image.
            dm: A numpy.ndarray, containing the true seismic image.
            pb: A tqdm.tqdm, containing the progress bar.
        """
        # Prediction error.
        model_error = ((dm_est - dm).norm()**2).item()

        # Print current objective value.
        pb.set_postfix(itr=f'{itr + 1}/{args.max_itr}',
                       obj=f'{obj}',
                       misfit=f'{model_error}')

        # Append current objective value and perdition error to log.
        self.obj_log.append(obj)
        self.error_log.append(model_error)

    def lr_decay(self, itr: int):
        """Reduces the learning rate of sampling sub-routine.

        Args:
            itr: An int, containing the current iteration number.
        """
        for param_group in self.sampler.param_groups:
            param_group['lr'] = self.lr_fn(itr)
