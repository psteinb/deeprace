"""
usage: deeprace run [options] [--] <models>

options:
    -h, --help                                 print this help message
    -O <mopts> --meta-options=<mopts>          hyper-parameters for training, e.g. batch_size
    -D <dpath> --datapath=<dpath>              path used for temporary storage, e.g. for the input data, checkpoints etc [default: datasets ]
    -e <neps> --nepochs=<neps>                 number of epochs to train [default: 0]
    -d <ds> --dataset=<ds>                     the dataset to use [default: cifar10]
    -f <dfrac> --datafraction=<dfrac>          fraction of the dataset to use, helpful for debugging/testing [default: 1.]
    -t <output> --timings=<output>             file to store the individual timings in [default: timings.tsv]
    -s <sep> --separator=<sep>                 seperator for the output data [default: \t]
    -c <cmt> --comment=<cmt>                   comment to add to the measurement
"""

from docopt import docopt
import os
import sys
import glob
import importlib
import logging

if importlib.find_loader:
    finder = importlib.find_loader
elif importlib.utils.find_spec:
    finder = importlib.utils.find_spec


def run_model(args):

    logging.info("dr_run received %s",args)

def main():

    args = docopt(__doc__)
    if args['--help']:
        print(__doc__)
        sys.exit(1)
    else:
        sys.exit(run_model(args))


if __name__ == '__main__':
    main()
