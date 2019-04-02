import logging
import os

import numpy as np

import keras.backend as K
from keras.models import Model


def model_size(inmodel):
    """Compute number of params in a inmodel (the actual number of floats)"""
    #stolen from https://stackoverflow.com/questions/35792278/how-to-find-number-of-parameters-of-a-keras-inmodel/35827171
    weights = inmodel.get_weights()

    return sum([np.prod(w.shape) for w in weights])

def to_disk(model, path):
    """ function to store the given model to disk, path should contain extension (which will later be replaced appropriately)
    returns a tuple of files written
    """

    weights_fname = os.path.splitext(path)[0]+'.h5'
    model_fname = os.path.splitext(path)[0] + ".json"

    with open(model_fname,"w") as mf:
        mf.write(model.to_json())
        mf.close()

    model.save_weights(weights_fname)

    logging.info("model saved in {0} and {1}".format(model_fname,weights_fname))

    return weights_fname, model_fname
