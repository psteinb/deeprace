"""
mweigert@mpi-cbg.de
"""

from __future__ import print_function, unicode_literals, absolute_import, division

import os
import numpy as np
from glob import glob
from tifffile import imread
from keras.optimizers import Adam
from care_model import resunet_model

def normalize(x, pmin=3, pmax=99.8, axis = None, clip=False):
    mi = np.percentile(x, pmin, axis = axis, keepdims=True).astype(np.float32)
    ma = np.percentile(x, pmax, axis = axis, keepdims=True).astype(np.float32)
    x = x.astype(np.float32)
    eps = 1.e-20
    y = (1. * x - mi) / (ma - mi+eps)
    if clip:
        y = np.clip(y, 0, 1)
    return y


def crop_center(x, shape=(256,256)):
    if any(s<d for s,d in zip(x.shape, shape)):
        raise ValueError("imgage to small!")

    ss = tuple(slice((s-d)//2,d+(s-d)//2) for s,d in zip(x.shape, shape))
    return x[ss]

def shuffle_inplace(*arrs):
    _state = np.random.get_state()
    for a in arrs:
        np.random.set_state(_state)
        np.random.shuffle(a)
    np.random.set_state(_state)


def resave_data(sigma = 400, root="data"):
    """ create pairs of (noisy,gt) pairs with gaussian noise of given value"""

    fnames = sorted(glob(os.path.join(root, "*.tif")))[:n_imgs]

    print("loading %s files" % len(fnames))
    for f in fnames:
        npa = imread(f)
        np.save(f.replace(".tif",".npy"),npa)


def create_data(sigma = 400, root="data", n_imgs = None, shape = (256,256)):
    """ create pairs of (noisy,gt) pairs with gaussian noise of given value"""

    fnames = sorted(glob(os.path.join(root, "*.tif")))[:n_imgs]

    print("loading %s files" % len(fnames))
    imgs = tuple(crop_center(imread(f),shape) for f in fnames)

    X = np.stack([normalize(im + sigma+sigma *  np.random.normal(0, 1, im.shape),1,99.8) for im in imgs])
    Y = np.stack([normalize(im,1,99.8) for im in imgs])


    shuffle_inplace(X,Y)

    return X[...,np.newaxis],Y[...,np.newaxis]



if __name__ == '__main__':


    X,Y = create_data()




    model = resunet_model(input_shape = (None,None,1),
                          last_activation = "linear",
                          n_depth=2,
                          n_filter_base=32,
                          n_row=3,
                          n_col=3,
                          n_conv_per_depth=2,
                          activation="relu",
                          )


    model.compile(optimizer=Adam(lr=0.0005), loss="mse")

    history = model.fit(X,Y,
                        epochs = 60,
                        batch_size = 16,
                        validation_split = 0.1,
                        shuffle = True
                        )




