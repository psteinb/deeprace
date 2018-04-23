import re
import numpy as np
import os
import logging
import math
import time

from distutils.util import strtobool
from .base import base_model


def compute_depth(n=3,version=1):
    value = 0
    if version == 1:
        value = n * 6 + 2
    elif version == 2:
        value = n * 9 + 2
    return value


def name(n=3,version=1):
    """ encrypt n and version into a standardized string """

    # Model name, depth and version
    value = 'resnet%dv%d' % (compute_depth(n,version), version)

    return value

def params_from_name(name):
    """ function that extracts a dictionary of parameters from a given name,
    e.g. resnet56v1 would result in { 'n' : 9, 'version' = 1 },
    this is the inverse of the 'name' function
    """

    found = re.findall('\d+',name)
    value = {'n' : None, 'version' : None}
    if not len(found) == 2:
        value['version'] = 1
    else:
        value['version'] = int(found[-1])

    depth = int(found[0])
    version = value['version']
    if version == 1:
        value['n'] = (depth - 2)//6
    if version == 2:
        value['n'] = (depth - 2)//9

    return value

class model(base_model):

    def __init__(self):
        self.num_classes =10
        self.n = 5
        self.version =1
        self.batch_size =32
        self.epochs =200
        self.data_augmentation =True
        self.subtract_pixel_mean =True
        self.checkpoint_epochs =False
        self.scratchspace = os.getcwd()
        self.backend = "keras"
        self.n_gpus = 1

    def provides(self):
        """ provide a list of strings which denote which models can be provided by this module """

        possible_values = [3,5,7,9,18,27]

        value = [ name(n=i,version=1) for i in possible_values ]

        possible_values.append(111)
        value.extend( [ name(n=i,version=2) for i in possible_values ] )

        #TODO: automate this
        backends = []
        from .keras_details import resnet_details as keras_resnet
        if keras_resnet.can_train():
            backends.append("keras")
        from .tf_details import resnet_details as tf_resnet
        if tf_resnet.can_train():
            backends.append("tensorflow")

        return value, backends

    def options(self):
        """ return a dictionary of options that can be provided to the train method besides the train and test dataset """

        return self.__dict__

    def data_loader(self, temp_path, dataset_name = "cifar10" ):

        #TODO: this if clause is non-sense, there must be a better way
        if "keras" in self.backend.lower():
            from .keras_details.resnet_details import data_loader
            return data_loader(temp_path, dataset_name)

        elif "tf" in self.backend.lower() or "tensorflow" in self.backend.lower():
            from .tf_details.resnet_details import data_loader
            return data_loader(temp_path, dataset_name)


    def train(self,train, test, datafraction = 1.):

        """setup the resnet and run the train function"""

        datafraction = float(datafraction)
        if datafraction > 1.0 or datafraction < 0:
            logging.error("resnet :: datafraction can only be [0,1]")

        #TODO: this if clause is non-sense, there must be a better way
        if "keras" in self.backend.lower():
            from .keras_details import resnet_details as keras_resnet
            return keras_resnet.train(train,test,datafraction,self.__dict__)
        if "tf" in self.backend.lower() or "tensorflow" in self.backend.lower():
            from .tf_details import resnet_details as tf_resnet
            return tf_resnet.train(train,test,datafraction,self.__dict__)

    def versions(self):

        value = ""

        if self.backend.lower().startswith("keras"):

            import keras
            from keras import backend as K

            value = "keras:{kver},backend:{bname}".format(kver=keras.__version__,bname=K.backend())

            if K.tf:
                value += ":" + K.tf.__version__
            else:
            #the following is untested!
                try:
                    if K.th:
                        value += ":" + K.th.__version__
                    else:
                        if K.cntk:
                            value += ":" + K.cntk.__version__
                except:
                    value += ":???"

        else:

            if self.backend.lower() == "tensorflow" or self.backend.lower() == "tf":
                import tensorflow as tf
                value = "tensorflow:{ver}".format(ver=tf.__version__)
            else:
                value = "unknown:0.0"

        return value
