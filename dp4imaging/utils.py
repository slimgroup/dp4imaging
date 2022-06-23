import typing
import argparse
import os
import h5py
import torch
import numpy as np


def get_velocity(vel_path: str = os.path.join(
    os.path.expanduser("~"), "data")) -> typing.Tuple[
        np.ndarray, np.ndarray, typing.Tuple, typing.Tuple, typing.Tuple]:
    """Downloads and returns a velocity model.

    Args:
        vel_path: A string containing path to the velocity model.

    Returns:
        m0: A numpy array containing the smooth background squared-slowness
            model.
        dm: A numpy array containing the true perturbation model (seismic
            image).
        spacing: A tuple containing the grid spacing of the model.
        shape: A tuple containing the shape of the model.
        origin: A tuple containing the origin of the model.
    """

    if not os.path.exists(vel_path):
        os.makedirs(vel_path)

    vel_file = os.path.join(vel_path, 'parihaka_model_high-freq.h5')

    if not os.path.exists(vel_file):
        os.system("wget https://www.dropbox.com/s/eouo2awl156vc94/"
                  "parihaka_model_high-freq.h5 -O" + vel_file)

    spacing = (25.0, 12.5)

    m0 = np.transpose(h5py.File(vel_file, 'r')['m0'][...])
    dm = np.transpose(h5py.File(vel_file, 'r')['dm'][...])
    shape = h5py.File(vel_file, 'r')['n'][...][::-1]
    origin = (0., 0.)

    return m0, dm, spacing, shape, origin


def setup_sample_file(args: argparse.Namespace, shape: typing.Tuple,
                      max_samples: int):
    """Setting up an HDF5 file to write samples.

    Args:
        args: An argparse.Namespace, containing command line arguments.
        shape: A tuple containing the shape of the model.
        max_samples: An integer containing the maximum number of samples that
            will be stored.
    """
    # Path to the file that samples will be written to.
    samples_file_path = os.path.join(args.checkpoint_path, 'samples.hdf5')

    # Samples file.
    samples_file = h5py.File(samples_file_path, 'w')
    dataset_shape = (max_samples, shape[0], shape[1])

    # HDF5 dataset for samples.
    samples_file.create_dataset('samples', dataset_shape, dtype=np.float32)

    # HDF5 dataset keeping track of number of samples saved so far.
    num_samples_saved = samples_file.create_dataset('num_samples', [1],
                                                    dtype=int)
    num_samples_saved[0] = 0
    samples_file.close()


def save_checkpoint(args: argparse.Namespace, obj_log: typing.List[float],
                    error_log: typing.List, G: torch.nn.Module,
                    z: torch.Tensor, samples_buffer: typing.List[np.ndarray]):
    """Saves the current state of the training/sampling process.

    Saves intermediate results, such as network parameters and training logs. It
    also takes in samples that are ready to be written to an HDF5 file.

    Args:
        args: An argparse.Namespace, containing command line arguments.
        obj_log: A list containing the objective function values.
        error_log: A list containing the error values.
        g: A torch.nn.Module containing the generator network.
        z: A torch.Tensor containing the fixed latent variable.
        samples_buffer: A list containing the samples that are ready to be
            written to file
    """

    if G is None:
        state_dict = None
    else:
        state_dict = G.state_dict()

    # Save intermediate network weights and training logs
    torch.save(
        {
            'obj_log': obj_log,
            'error_log': error_log,
            'model_state_dict': state_dict,
            'z': z
        }, os.path.join(args.checkpoint_path, 'checkpoint.pth'))

    # Save smaples in buffer
    if len(samples_buffer) > 0:
        samples_file_path = os.path.join(args.checkpoint_path, 'samples.hdf5')
        samples_file = h5py.File(samples_file_path, 'r+')
        num_samples_saved = samples_file["num_samples"]

        shape = samples_file['samples'].shape[1:]
        samples_buffer = np.array(samples_buffer)
        samples_buffer = samples_buffer.reshape(-1, shape[0], shape[1])

        samples_file['samples'][num_samples_saved[0]:(num_samples_saved[0] +
                                                      samples_buffer.shape[0]),
                                ...] = samples_buffer

        num_samples_saved[0] += samples_buffer.shape[0]
        samples_file.close()


def decay_fn(args):
    """
    Learning rate scheduler
    """
    if args.lr == args.lr_final:

        def lr_fn(t):
            return args.lr
    else:
        lag = 0
        max_itr_lag = args.max_itr - lag
        b = max_itr_lag / ((args.lr_final / args.lr)**(1 / args.gamma) - 1.0)
        a = args.lr / (b**args.gamma)

        def lr_fn(t, a=a, b=b, gamma=args.gamma, lag=lag):
            if t < lag:
                return args.lr
            else:
                return a * ((b + t)**gamma)

    return lr_fn
