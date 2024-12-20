<h1 align="center">Deep Bayesian inference for seismic imaging with tasks</h1>

Code to partially reproduce results in [Deep Bayesian inference for
seismic imaging with tasks](https://arxiv.org/abs/2110.04825).

## Installation

For further development and to run the examples, clone the repository
and install the package in editable mode. **Make sure to adapt CUDA
version in `setup.cfg` to the one installed on your system by specifying
in as `torch-cuda==XY.Z`.**

```bash
# Create a new conda environment.
conda create --name dp4imaging "python<=3.11"
conda activate dp4imaging

# Clone the repository and install the package in editable mode.
git clone https://github.com/slimgroup/dp4imaging
cd dp4imaging/
pip install -e .
```

After the above steps, you can run the example scripts by just
activating the environment, i.e., `conda activate dp4imaging`, the
following times.

## Questions

Please contact alisk@gatech.edu for questions.

## Author

Ali Siahkoohi
