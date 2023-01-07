import argparse


def init_param():
    parser = argparse.ArgumentParser('LandslideDataSet')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10)

    parser.add_argument('--verbose_step', type=int, default=10)

    return parser.parse_args()


arg = init_param()
