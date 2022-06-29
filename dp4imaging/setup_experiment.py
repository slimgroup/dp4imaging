from devito import configuration
from examples.seismic import Model, AcquisitionGeometry, TimeAxis, RickerSource
from examples.seismic.acoustic import AcousticWaveSolver
import h5py
import numpy as np
import os
from scipy import signal
import torch
from tqdm import tqdm

from dp4imaging.devito_layer import ForwardBorn
from dp4imaging.project_path import datadir
from dp4imaging.utils import get_velocity
import logging

configuration['log-level'] = 'WARNING'
logging.basicConfig(level=logging.INFO)


class ForwardBornLayer(torch.nn.Module):

    def __init__(self, model, src, geometry, device):
        super(ForwardBornLayer, self).__init__()
        self.forward_born = ForwardBorn()
        self.model = model
        self.src = src
        self.geometry = geometry
        self.device = device
        self.solver = AcousticWaveSolver(self.model,
                                         self.geometry,
                                         space_order=16)

    def forward(self, x):
        return self.forward_born.apply(x, self.src, self.model, self.geometry,
                                       self.solver, self.device)


class SeismicSetup(object):
    """Creating Devito wrapper and utilities.
    """

    def __init__(self, device, sigma, sim_source=False):
        super(SeismicSetup, self).__init__()

        self.device = device
        self.tn = 1500.0
        self.f0 = 0.030
        self.nsrc = 205
        self.nrec = 410
        self.sigma = sigma
        self.sim_source = sim_source
        if self.sim_source:
            self.nsimsrc = self.nsrc
            self.src_enc = np.random.randn(self.nsrc, self.nsimsrc)
            self.src_enc *= 1.0 / np.sqrt(self.nsimsrc)
        else:
            self.nsimsrc = 1
            self.src_enc = None

        self.setup_model()
        self.forward_op = self.wrap_op()

    def mute_op(self, dm, end=10, length=5):
        start = end - length
        damp = torch.zeros([dm.shape[-2], dm.shape[-1]], device=dm.device)
        damp[..., end:] = 1.
        damp[..., start:end] = (1. + torch.sin(
            (np.pi / 2.0 * torch.arange(0, length)) / (length))) / 2.
        return damp * dm

    def setup_model(self):
        """
        Load velocity and setup Devito model
        """
        self.m0, dm, self.spacing, shape, origin = get_velocity()
        self.dm = torch.from_numpy(dm).unsqueeze(0).unsqueeze(0)
        self.model = Model(space_order=16,
                           vp=1.0 / np.sqrt(self.m0),
                           origin=origin,
                           shape=self.m0.shape,
                           dtype=np.float32,
                           spacing=self.spacing,
                           nbl=40,
                           bcs='damp')

    def wrap_op(self):
        """
        Initialize the wrapped Devito operator
        """
        dt = self.model.critical_dt
        time_range = TimeAxis(start=0.0, stop=self.tn, step=dt)
        self.nt = time_range.num

        rec_coordinates = np.empty((self.nrec, len(self.model.spacing)),
                                   dtype=np.float32)
        rec_coordinates[:, 0] = np.linspace(0,
                                            self.model.domain_size[0],
                                            num=self.nrec)
        rec_coordinates[:, 1] = 2.0 * self.model.spacing[1]

        # Source geometry
        src = RickerSource(name='src',
                           grid=self.model.grid,
                           f0=self.f0,
                           time_range=time_range,
                           npoint=self.nsimsrc)
        self.wavelet = np.array(src.data[:, 0:1])
        src.coordinates.data[:, 0] = np.linspace(0,
                                                 self.model.domain_size[0],
                                                 num=self.nsimsrc)
        src.coordinates.data[:, -1] = 2.0 * self.model.spacing[1]

        # Create geometry
        geometry = AcquisitionGeometry(self.model,
                                       rec_coordinates,
                                       src.coordinates.data,
                                       t0=0.0,
                                       tn=self.tn,
                                       src_type='Ricker',
                                       f0=self.f0)
        return ForwardBornLayer(self.model, src, geometry, self.device)

    def create_op(self, src_idx=0):
        """
        Update the operator with the source index
        """
        if not self.sim_source:
            self.forward_op.src.coordinates_data[:,
                                                 0] = src_idx * self.model.spacing[
                                                     0]
        else:
            self.forward_op.src.data[:] = self.wavelet
            self.forward_op.src.data[:] *= self.src_enc[src_idx, :]

        def J(x):
            return self.forward_op(self.mute_op(x))

        return J

    def create_sim_src_data(self):
        """Create simultaneous source data.
        """
        d_lin = simulate_sequential_data(self.sigma)[...]

        sim_data_path = os.path.join(datadir("observed_data"),
                                     'sim_src_data.h5')
        sim_data_file = h5py.File(sim_data_path, 'w')
        sim_data = sim_data_file.create_dataset(
            'data', [self.nsrc, self.nt, self.nrec], dtype=np.float32)

        logging.info('Creating simultaneous source experiments')
        for i in tqdm(range(self.nsrc)):
            for j in range(self.nsimsrc):
                sim_data[i, ...] += self.src_enc[i, j] * d_lin[j, ...]

        sim_data_file.close()

        return h5py.File(sim_data_path, 'r')["data"]


def simulate_sequential_data(sigma):
    """Simulates simultaneous source data and adds noise.
    """

    imaging_setup = SeismicSetup('cpu', sigma, sim_source=False)
    dm = imaging_setup.dm

    data_path = os.path.join(datadir("observed_data"), 'observed_data.h5')
    data_file = h5py.File(data_path, 'w')
    d_lin = data_file.create_dataset(
        'data', [imaging_setup.nsrc, imaging_setup.nt, imaging_setup.nrec],
        dtype=np.float32)

    logging.info('Simulating observed data')
    for idx in tqdm(range(imaging_setup.nsrc)):
        forward_op = imaging_setup.create_op(src_idx=idx)
        d = forward_op(dm)
        d_lin[idx, :, :] = d.detach().numpy().astype(np.float32)

    # Add noise to get observed data (-8.7466 dB)
    kernel = np.outer(imaging_setup.wavelet[:60], imaging_setup.wavelet[:60])
    for j in range(d_lin.shape[0]):
        e = sigma * np.random.randn(*d_lin.shape[1:])
        e = signal.fftconvolve(e, kernel, mode='same')
        e *= 1 / np.linalg.norm(e)
        e *= np.sqrt(np.prod(e.shape)) * sigma
        d_lin[j, ...] += e

    data_file.close()

    return h5py.File(data_path, 'r')["data"]
