from __future__ import print_function, unicode_literals, absolute_import, division
import os
import numpy as np
from glob import glob
import logging

def resave_to_chunks(root="data", n_imgs = None, output_stem="care_denoise2d_all"):
    """ store tif images found in <root> and resave them into a compressed numpy format under <root>/<output_stem>.npz """

    from tifffile import imread

    fnames = sorted(glob(os.path.join(root, "*.tif")))[:n_imgs]
    loaded = {}

    logging.info("loading %s files" % len(fnames))
    for f in fnames:
        stem = os.path.split(f)[-1]
        loaded[stem] = (imread(f))

    #stacked = np.stack(loaded)
    dst = os.path.join(root, output_stem)
    np.savez_compressed(dst, **loaded)
    produced = glob(dst+".np*")
    if len(produced) > 0:
        logging.info("stored %i files in %s" % (len(fnames), produced))
        return produced[0]
    else:
        logging.warning("unable to store %i files in %s (file not found)" % (len(fnames), produced))
        return None

def load_chunk(fname=None, n_imgs = None):
    """ load images from <fname> which are expected to be compressed numpy arrays and return a list of arrays found """

    if not os.path.exists(fname):
        logging.error("input file %s does not exist",fname)
        return None

    loaded = np.load(fname)
    logging.debug("[load_chunk] loaded %i files from %s" % (len(loaded.files), fname))
    return [ loaded[item] for item in loaded.files ]
