"""CIFAR10 small images classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import os
import urllib.request
import shutil
from datasets.care_denoise2d_data import create_data_from_chunks
import logging

def load_data(temp_dir='datasets'):
    """Loads CIFAR10 dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    origin = 'https://idisk.mpi-cbg.de/~steinbac/deeprace/care_denoise2d_all.npz'
    fname = 'care_denoise2d_all.npz'

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    stored_loc = os.path.join(temp_dir,fname)
    X,Y = None,None
    with urllib.request.urlopen(origin) as response, open(stored_loc, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        out_file.close()
        logging.debug("downloaded %s to %s (size %d)",origin,stored_loc,os.stat(stored_loc).st_size)
        X,Y = create_data_from_chunks(chunk_loc=stored_loc)
        logging.debug("unpacked %s and %s", X.shape,Y.shape)


    # if K.image_data_format() == 'channels_last':
    #     x_train = x_train.transpose(0, 2, 3, 1)
    #     x_test = x_test.transpose(0, 2, 3, 1)

    return (X, Y)
