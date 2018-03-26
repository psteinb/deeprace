import keras.backend as K
from keras.models import Model

import numpy as np

def model_size(inmodel):
    """Compute number of params in a inmodel (the actual number of floats)"""
    #stolen from https://stackoverflow.com/questions/35792278/how-to-find-number-of-parameters-of-a-keras-inmodel/35827171
    weights = inmodel.get_weights()

    return sum([np.prod(w.shape) for w in weights])
