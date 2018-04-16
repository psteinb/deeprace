import logging
import math
import numpy as np
import os
import time
import importlib

def can_train():

    tf_found = importlib.util.find_spec('tensorflow')

    if tf_found:
        return True
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
    
    model = cf10(resnet_size=optsdict["n"], num_classes=optsdict["n"],version=optsdict["version"])
    return None, None, { 'num_weights' : None }
