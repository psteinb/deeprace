import logging
import math
import os
import sys
import time
import importlib

import tarfile
from six.moves import urllib

#thanks to https://stackoverflow.com/a/11887825
def versiontuple(v, version_index = -1):
    """ convert a version string to a tuple of integers
    argument <v> is the version string, <version_index> refers o how many '.' splitted shall be returned

    example:
    versiontuple("1.7.0") -> tuple([1,7,0])
    versiontuple("1.7.0", 2) -> tuple([1,7])
    versiontuple("1.7.0", 1) -> tuple([1])

    """

    temp_list = map(int, (v.split(".")))
    return tuple(temp_list)[:version_index]

def can_train():

    tf_found = importlib.util.find_spec('tensorflow')

    if tf_found:
        #thanks to https://github.com/pydata/pandas/issues/2841
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)

        from tensorflow import __version__ as tfv
        required = "1.7.0"

        #only require major and minor release number as the patch number may contain 'rc' etc
        if versiontuple(tfv,2) >= versiontuple(required,2):
            return True
        else:
            return False
    else:
        return False

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

    parser = run_loop.ResnetArgParser()
    
    model_dir = os.path.join(opts['scratchspace'],'model')
    model_dir = os.path.abspath(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    data_location = os.path.join(opts['datapath']# , 'cifar-10-batches-bin'
    )
    parser.set_defaults(data_dir=data_location,
                        model_dir=model_dir,
                        resnet_size=compute_depth(opts["n"],opts["version"]),
                        train_epochs=opts['epochs'],
                        epochs_between_evals=1,
                        version=int(opts["version"]),
                        batch_size=opts['batch_size'],
                        multi_gpu = opts["n_gpus"] > 1)

    flags = parser.parse_args(args=[])
    logging.info('handing over \n >> %s \n >>  %s',flags,opts)
    results = run_loop.resnet_main(flags, cfmain.cifar10_model_fn, cfmain.input_fn, opts)

    return None, None, { 'num_weights' : None }
