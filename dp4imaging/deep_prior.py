"""Implementation the deep prior network.

This implementation is based on the original implementation of the deep
priors in https://github.com/DmitryUlyanov/deep-image-prior.

Typical usage example:

g = DeepPrior(shape)
z = torch.randn(g.get_latent_shape())
g(z)
"""

from math import floor, ceil
import numpy as np
import torch
from typing import List, Sequence, Tuple, Union


def add_module(self, module: torch.nn.Module):
    """Define the `add_module` method to use module number as module name.

    Args:
        module: A torch.nn.Module object.
    """
    self.add_module(str(len(self) + 1), module)


# Redefine the `add` method to use modules number as module name.
torch.nn.Module.add = add_module


class Concat(torch.nn.Module):
    """A concatenation module (layer).

    This module defines a concatenation "layer" that can be used to concatenate
    multiple activations at a given dimension.

    Attributes:
        dim: An integer indicating the dimension to concatenate the activations
            at.
    """

    def __init__(self, dim: int, *args: torch.nn.Module):
        """Initializes a Concat module.

        Args:
            dim: An integer indicating the dimension to concatenate the
                activations at.
            args: A list of torch.nn.Module objects, containing the modules to
                concatenate their activations.
        """
        super().__init__()
        self.dim = dim
        for module in args:
            self.add(module)

    def __len__(self) -> int:
        """Define the `__len__` method that returns the number of modules.

        Returns:
            An integer, the number of modules.
        """
        return len(self._modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenates the activations.

        This function first evaluates the activations of all the modules
        provided as input during initialization. Next, it concatenates the
        activations at the dimension specified during initialization.

        Args:
            x: A torch.Tensor object, containing the input to the `Concat`
                layer.

        Returns:
            A torch.Tensor containing the concatenated activations.
        """
        inputs = []
        for module in self._modules.values():
            inputs.append(module(x))

        inputs_shapes2 = [x_.shape[2] for x_ in inputs]
        inputs_shapes3 = [x_.shape[3] for x_ in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and \
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2:diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)


class View(torch.nn.Module):
    """A module to create a view of an existing torch.Tensor (avoid copying).

    Attributes:
        shape: A tuple containing the desired shape of the view.
    """

    def __init__(self, *shape: int):
        """Initializes a Concat module.

        Args:
            shape: A tuple containing the desired shape of the view.
        """
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Creates a view of an input.

        Args:
            x: A torch.Tensor object.

        Returns:
            A torch.Tensor containing the view of the input with given
                dimensions.
        """
        return x.view(*self.shape)


def pad_and_conv(c_in: int,
                 c_out: int,
                 kernel_size: int,
                 stride: int = 1,
                 bias: bool = True) -> torch.nn.Module:
    """A utility function to define a 2D convolutional layer with padding.

    Args:
        c_in: An integer, the number of input channels.
        c_out: An integer, the number of output channels.
        kernel_size: An integer, the size of the convolutional kernel.
        stride: An optional integer, the stride of the convolutional kernel.
        bias: An optional boolean, indicating whether to use bias.

    Returns:
        A torch.nn.Conv2d object.
    """
    # Amount to pad.
    to_pad = int((kernel_size - 1) / 2)
    # Reflect padding.
    padding_layer = torch.nn.ReflectionPad2d(to_pad)
    # Convolutional layer.
    conv_layer = torch.nn.Conv2d(c_in,
                                 c_out,
                                 kernel_size,
                                 stride,
                                 bias=bias,
                                 padding=0)
    return torch.nn.Sequential(*[padding_layer, conv_layer])


class Crop(torch.nn.Module):
    """A module to crop in the input tensor.

    This module computes the padding dimensions and cropping amount needed to
    ensure same input and output deep prior network sizes.

    Attributes:
        input_shape: A 4-component list or tuple containing the input tensor
            shape.
        crop_shape: A list containing per dimension cropping shape.
        latent_shape: A tuple containing the latent variable shape.
    """

    def __init__(self, input_shape: Sequence, n_scales: int):
        """Initializes a Crop "layer".

        Args:
            args: An argparse.Namespace, containing command line arguments.
        """
        super().__init__()

        self.input_shape = input_shape
        self.crop_shape, self.latent_shape = self.find_cropping_shape(n_scales)

    def find_cropping_shape(self, n_scales: int) -> Tuple[List, List]:
        """Finds cropping shape.

        Cropping shape such that the deep prior output shape is the same size as
        `input_shape`.

        Args:
            n_scales: An integer, the number of scales.

        Returns:
            crop_shape: A list containing the cropping shape.
            latent_shape: A list containing the latent variable shape.

        """
        # Downsampling factor.
        downsample_factor = 2**n_scales

        # Total cropping shape.
        total_crop_shape = [
            min(
                self.input_shape[i + 2] % downsample_factor,
                downsample_factor -
                self.input_shape[i + 2] % downsample_factor) for i in range(2)
        ]

        # Per dimension cropping shape.
        crop_shape = np.zeros([2, 2], dtype=int)
        crop_shape[0, 0] = int(floor(total_crop_shape[0] / 2.))
        crop_shape[0, 1] = int(ceil(total_crop_shape[0] / 2.))
        crop_shape[1, 0] = int(floor(total_crop_shape[1] / 2.))
        crop_shape[1, 1] = int(ceil(total_crop_shape[1] / 2.))

        # Latent variable shape.
        latent_shape = [
            self.input_shape[i + 2] + total_crop_shape[i] for i in range(2)
        ]

        return crop_shape, latent_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Crops in the input tensor according to the cropping shape.

        Args:
            x: A torch.Tensor object.

        Returns:
            A torch.Tensor containing the cropped input tensor.
        """
        if (self.input_shape[2] == self.latent_shape[0]
                and self.input_shape[3] == self.latent_shape[1]):
            y = x
        elif self.input_shape[2] == self.latent_shape[0]:
            y = x[:, :, :, self.crop_shape[1, 0]:-self.crop_shape[1, 1]]
        elif self.input_shape[3] == self.latent_shape[1]:
            y = x[:, :, self.crop_shape[0, 0]:-self.crop_shape[0, 1], :]
        else:
            y = x[:, :, self.crop_shape[0, 0]:-self.crop_shape[0, 1],
                  self.crop_shape[1, 0]:-self.crop_shape[1, 1]]
        return y


class Scale(torch.nn.Module):
    """A scaling module (layer).

    This module simply scales the its input tensor with a fixed value (float).

    Attributes:
        scaling_factor: A float, the scaling factor.
    """

    def __init__(self, scaling_factor: float):
        """Initializes a Scale module.

        Args:
            scaling_factor: A float, the scaling factor.
        """
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scales the input tensor by `scaling_factor`.

        Args:
            x: A torch.Tensor object.

        Returns:
            A torch.Tensor scaled by `scaling_factor`
        """
        return x * self.scaling_factor


class DeepPrior(torch.nn.Module):
    """Deep prior network implementation.

    The CNN architecture proposed by Lempitsky et al. (2018) is a variation of
    the widely used U-net architecture, as described by Ronneberger et al.
    (2015). The U-net architecture is composed of an encoder and a decoder
    module, where information, i.e., intermediate values, from the encoder
    module is also passed to the decoder through skip connections.

    The following are the major differences between the deep prior architecture
    (Lempitsky et al., 2018) and the U-net architecture. Contrary to U-net, the
    convolutional layers in the decoding module of the deep prior architecture
    do not increase the dimensionality of the intermediate values. Rather,
    dimensionality-preserving convolutional layers, that is, stride-one
    convolutions, are augmented with user-defined interpolation schemes to
    achieve upsampling. This enables the degree of smoothness in the image space
    to be controlled by the interpolation kernel. Another difference worth
    noting is the way intermediate values from the encoding phase are
    incorporated into the decoding module through skip connections. In the deep
    prior architecture, the intermediate values in the encoding phase are passed
    through an additional convolutional layer before being fed into the decoder
    module.

    This implementation is based on the original implementation of the deep
    priors in https://github.com/DmitryUlyanov/deep-image-prior. For
    schematic representation of the architecture with the following default
    parameters see Figure 15 in https://arxiv.org/abs/2110.04825.

    Attributes:
        ch_in: An integer, the number of input channels.
        model: A torch.nn.Sequential object, containing the deep prior network.

    """

    def __init__(self,
                 input_shape: Sequence,
                 ch_in: int = 3,
                 ch_out: int = 1,
                 ch_encoding: Union[int, Tuple, List] = (16, 32, 64),
                 ch_decoding: Union[int, Tuple, List] = (16, 32, 64),
                 ch_skip: Union[int, Tuple, List] = (0, 0, 64),
                 kernel_size_encoding: Union[int, Tuple, List] = 5,
                 kernel_size_decoding: Union[int, Tuple, List] = 5,
                 kernel_size_skip: int = 1,
                 bias: bool = True,
                 upsample_mode: str = 'nearest',
                 activation_fn: torch.nn.Module = torch.nn.LeakyReLU(
                     0.2, inplace=True),
                 scaling_factor: float = 1.0 / 40.0):
        """Initializes a DeepPrior object.

        Args:
        input_shape: A 4-component list or tuple containing the input tensor
            shape.
        ch_in: An optional integer, the number of input (latent variable)
            channels.
        ch_out: An optional integer, the number of output (image) channels.
        ch_encoding: An optional integer, list, or tuple of the number channels
            in the encoder module.
        ch_decoding: An optional integer, list, or tuple of the number channels
            in the decoding module.
        ch_skip: An optional integer, list, or tuple of the number channels
            in the skip connections.
        kernel_size_encoding: An optional integer, list, or tuple of the
            kernel sizes in the encoder module.
        kernel_size_decoding: An optional integer, list, or tuple of the
            kernel sizes in the decoding module.
        kernel_size_skip: An optional integer for the kernel size in the
            skip connections.
        bias: An optional boolean, whether to use bias in the convolutional
            layers.
        upsample_model: An optional string, the upsampling model to use in the
            decoder.
        activation_fn: An optional torch.nn.Module, the activation function.
        scaling_factor: An optional float, the scaling factor to multiply the
            output of the network with.
        """
        super().__init__()

        if len(ch_encoding) != len(ch_decoding):
            ValueError('`ch_encoding` must be of same length as '
                       '`ch_decoding`')
        if len(ch_encoding) != len(ch_skip):
            ValueError('`ch_encoding` must be of same length as '
                       '`ch_skip`')
        if len(ch_decoding) != len(ch_skip):
            ValueError('`ch_decoding` must be of same length as '
                       '`ch_skip`')

        self.ch_in = ch_in
        n_scales = len(ch_encoding)

        if not isinstance(upsample_mode, (list, tuple)):
            upsample_mode = [upsample_mode] * n_scales
        if not isinstance(kernel_size_encoding, (list, tuple)):
            kernel_size_encoding = [kernel_size_encoding] * n_scales
        if not isinstance(kernel_size_decoding, (list, tuple)):
            kernel_size_decoding = [kernel_size_decoding] * n_scales

        self.model = torch.nn.Sequential()
        model = self.model

        for i in range(len(ch_encoding)):
            model_ = torch.nn.Sequential()
            skip = torch.nn.Sequential()

            if ch_skip[i] != 0:
                model.add(Concat(1, skip, model_))
            else:
                model.add(model_)

            model.add(
                torch.nn.BatchNorm2d(ch_skip[i] +
                                     (ch_decoding[i + 1] if i <
                                      (n_scales - 1) else ch_encoding[i])))

            if ch_skip[i] != 0:
                skip.add(
                    pad_and_conv(
                        ch_in,
                        ch_skip[i],
                        kernel_size_skip,
                        bias=bias,
                    ))
                skip.add(torch.nn.BatchNorm2d(ch_skip[i]))
                skip.add(activation_fn)

            model_.add(
                pad_and_conv(
                    ch_in,
                    ch_encoding[i],
                    kernel_size_encoding[i],
                    2,
                    bias=bias,
                ))
            model_.add(torch.nn.BatchNorm2d(ch_encoding[i]))
            model_.add(activation_fn)

            model_.add(
                pad_and_conv(
                    ch_encoding[i],
                    ch_encoding[i],
                    kernel_size_encoding[i],
                    bias=bias,
                ))
            model_.add(torch.nn.BatchNorm2d(ch_encoding[i]))
            model_.add(activation_fn)

            deeper_main = torch.nn.Sequential()
            if i == len(ch_encoding) - 1:
                k = ch_encoding[i]
            else:
                model_.add(deeper_main)
                k = ch_decoding[i + 1]

            if not upsample_mode[i] == 'nearest':
                model_.add(
                    torch.nn.Upsample(scale_factor=2,
                                      mode=upsample_mode[i],
                                      align_corners=True))
            else:
                model_.add(
                    torch.nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

            model.add(
                pad_and_conv(
                    ch_skip[i] + k,
                    ch_decoding[i],
                    kernel_size_decoding[i],
                    1,
                    bias=bias,
                ))
            model.add(torch.nn.BatchNorm2d(ch_decoding[i]))
            model.add(activation_fn)

            model.add(
                pad_and_conv(
                    ch_decoding[i],
                    ch_decoding[i],
                    1,
                    bias=bias,
                ))
            model.add(torch.nn.BatchNorm2d(ch_decoding[i]))
            model.add(activation_fn)

            ch_in = ch_encoding[i]
            model = deeper_main

        self.model.add(pad_and_conv(
            ch_decoding[0],
            ch_out,
            1,
            bias=bias,
        ))
        self.model.add(Crop(input_shape, n_scales))
        self.model.add(Scale(scaling_factor))

    def get_latent_shape(self) -> List[int]:
        """Get the latent variable shape that results in correct output shape.

        Returns:
            A list of integers containing the latent variable shape.
        """
        return [1, self.ch_in] + self.model[-2].latent_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DeepPrior network.

        Args:
            x: A torch.Tensor containing the input latent variable.

        Returns:
            A torch.Tensor containing the output tensor (image).
        """
        return self.model(x)
