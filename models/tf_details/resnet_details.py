import argparse
import logging
import math
import os
import sys
import time
import importlib

import tarfile
from six.moves import urllib
from ..tools.utils import versiontuple
from .boilerplate import ResnetArgParser

def can_train():

    tf_found = importlib.util.find_spec('tensorflow')
    available_backends = []
    if tf_found:
        #thanks to https://github.com/pydata/pandas/issues/2841
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)

        from tensorflow import __version__ as tfv
        required = "1.6.0"

        #only require major and minor release number as the patch number may contain 'rc' etc
        if versiontuple(tfv,2) >= versiontuple(required,2):
            available_backends.append("tensorflow")

    return available_backends

def data_loader(path, dsname = "cifar10"):

    DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(path, filename)

    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, 100.0 * count * block_size / total_size))
        sys.stdout.flush()

      filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
      print('Nothing to do, %s exists' % filepath)

    tarfile.open(filepath, 'r:gz').extractall(path)

    #TODO: add some logic to see if all samples have been extracted

    return None, None, 5e4, 1e4


def compute_depth(n=3,version=1):
    value = 0
    if version == 1:
        value = n * 6 + 2
    elif version == 2:
        value = n * 9 + 2
    return value


def train(train, test, datafraction, opts):

    """setup the resnet and run the train function, train and test will be None here as reading the files from disk needs to be part of the compute graph AFAIK """

    from . import cifar10_main as cfmain
    from . import resnet_run_loop as run_loop
    opts["datafraction"] = datafraction

    model_dir = os.path.join(opts['scratchspace'],'cifar10_model')
    model_dir = os.path.abspath(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data_location = os.path.join(opts['datapath']# , 'cifar-10-batches-bin'
    )

    opts["ntrain"] = 50000
    opts["ntest"] = 10000

    parser = ResnetArgParser()
    parser.set_defaults(data_dir=data_location,
                        model_dir=model_dir,
                        resnet_size=compute_depth(opts["n"],opts["version"]),
                        train_epochs=opts['epochs'],
                        epochs_between_evals=1,
                        version=int(opts["version"]),
                        batch_size=opts['batch_size'],
                        multi_gpu = int(opts["n_gpus"]) > 1)

    flags = parser.parse_args(args=[])
    logging.info('handing over \n >> %s \n >>  %s',flags,opts)
    history, timings = run_loop.resnet_main(flags, cfmain.cifar10_model_fn, cfmain.input_fn, opts)

    if not opts['checkpoint_epochs']:
        logging.info("unable to ensure pure no-checkpoint behavior with resnet in pure tensorflow, removing result directory")
        import shutil
        shutil.rmtree(model_dir)

    return history, timings, { 'num_weights' : None }

def infer(data, num_inferences, optsdict):

    """ perform <num_inferences> on the given data """
    from . import cifar10_main as cfmain
    from . import resnet_run_loop as run_loop
