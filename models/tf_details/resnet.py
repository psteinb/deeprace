import logging
import math
import numpy as np
import os
import time
import importlib

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

def compute_depth(n=3,version=1):
    value = 0
    if version == 1:
        value = n * 6 + 2
    elif version == 2:
        value = n * 9 + 2
    return value

def train(train, test, datafraction, optsdict):

    """setup the resnet and run the train function"""

    from .cifar10 import Cifar10Model as cf10

    depth = compute_depth(optsdict["n"],optsdict["version"])
    model = cf10(resnet_size=depth, num_classes=optsdict["num_classes"],version=optsdict["version"])

    return None, None, { 'num_weights' : None }
