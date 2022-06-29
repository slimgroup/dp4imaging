"""Script to run imaging and uncertainty quantification with deep priors.

Typical usage example:

python run_imaging.py --phase sample --cuda 1
"""

import argparse
from scripts.deep_prior_imaging import DeepPriorImaging

parser = argparse.ArgumentParser(description='')
parser.add_argument('--max_itr',
                    dest='max_itr',
                    type=int,
                    default=10001,
                    help='maximum number of updates to weights')
parser.add_argument('--sigma',
                    dest='sigma',
                    type=float,
                    default=0.25,
                    help='noise standard variation')
parser.add_argument('--lr',
                    dest='lr',
                    type=float,
                    default=0.01,
                    help='initial learning rate for pSGLD')
parser.add_argument('--lr_final',
                    dest='lr_final',
                    type=float,
                    default=0.005,
                    help='final learning rate for pSGLD')
parser.add_argument('--wd',
                    dest='wd',
                    type=float,
                    default=30.0,
                    help='weight decay coefficient')
parser.add_argument('--gamma',
                    dest='gamma',
                    type=float,
                    default=-0.3333,
                    help='learning rate decay rate')
parser.add_argument('--experiment',
                    dest='experiment',
                    default='deep-bayesian-inference',
                    help='experiment name')
parser.add_argument('--cuda',
                    dest='cuda',
                    type=int,
                    default=0,
                    help='set to 1 for running on GPU, 0 for CPU')
parser.add_argument('--phase',
                    dest='phase',
                    default='sample',
                    help='sample or inference')
args = parser.parse_args()


def main():
    """Calls DeepPriorImaging with the input command line arguments.
    """

    # Ensure that noise standard deviation is greater than zero.
    if args.sigma <= 0.0:
        ValueError(f'`sigma` must be greater than zero but is {args.sigma}')

    # Ensure that initial learning rate is not less than final learning rate.
    if args.lr < args.lr_final:
        ValueError(f'`lr` ({args.lr}) must not be smaller than '
                   f'`lr_final ({args.lr_final}).')

    # Experiment name according to input arguments.
    args.experiment = (f'{args.experiment}_max_itr-{args.max_itr}_sigma'
                       f'-{args.sigma}_lr-{args.lr}_lr_final-{args.lr_final}_'
                       f'wd-{args.wd}_gamma-{args.gamma}')

    if args.phase == 'sample':
        imaging_instance = DeepPriorImaging(args)
        imaging_instance.sample(args)


if __name__ == '__main__':
    main()
