from __future__ import print_function, unicode_literals, absolute_import, division
import os
import numpy as np
from glob import glob
from tifffile import imread


def resave_to_chunks(root="data", n_imgs = None, output_stem="care_denoise2d_all"):
    """ create pairs of (noisy,gt) pairs with gaussian noise of given value"""

    fnames = sorted(glob(os.path.join(root, "*.tif")))[:n_imgs]
    loaded = {}

    print("loading %s files" % len(fnames))
    for f in fnames:
        stem = os.path.split(f)[-1]
        loaded[stem] = (imread(f))

    #stacked = np.stack(loaded)
    dst = os.path.join(root, output_stem)
    np.savez_compressed(dst, **loaded)
    produced = glob(dst+".np*")
    if len(produced) > 0:
        print("stored %i files in %s" % (len(fnames), produced))
        return produced[0]
    else:
        print("unable to store %i files in %s (file not found)" % (len(fnames), produced))
        return None

def load_from_chunks(fname=None, n_imgs = None):
    """ create pairs of (noisy,gt) pairs with gaussian noise of given value"""

    if not os.path.exists(fname):
        print("input file %s does not exist",fname)
        return None

    loaded = np.load(fname)
    print("loaded %i files from %s" % (len(loaded.files), fname))
    return [ loaded[item] for item in loaded.files ]
