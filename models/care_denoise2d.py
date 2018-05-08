import re
import numpy as np
import os
import logging
import math
import time
import glob

from distutils.util import strtobool
from .base import base_model


def name(n=2):
    """ encrypt n and version into a standardized string """

    # Model name, depth and version
    value = 'care_denoise2d_depth%d' % (n)

    return value

def params_from_name(name):
    """ function that extracts a dictionary of parameters from a given name,
    e.g. care_denoise2d_depth2 would result in { 'n_depth' : 2 },
    this is the inverse of the 'name' function
    """

    found = re.findall('\d+',name)
    value = {'n_depth' : None}
    if not len(found) == 2:
        value['n_depth'] = 2
    else:
        value['n_depth'] = int(found[-1])

    return value

class model(base_model):

    def __init__(self):

        self.n = 2
        self.version =1
        self.filter_base = 32
        self.n_row =True

        self.batch_size =32
        self.epochs = 60
        self.checkpoint_epochs =False
        self.scratchspace = os.getcwd()
        self.backend = "keras"
        self.n_gpus = 1

    def provides(self):
        """ provide a list of strings which denote which models can be provided by this module """

        possible_values = [2]

        value = [ name(n=i) for i in possible_values ]


        #TODO: automate this
        backends = []
        from .keras_details import care_denoise2d_details as keras_net
        if keras_net.can_train() != []:
            backends.extend(keras_net.can_train())

        from .keras_details import tfkeras_care_denoise2d_details as tfkeras_net
        if tfkeras_net.can_train() != []:
            backends.extend(tfkeras_net.can_train())

        return value, backends

    def options(self):
        """ return a dictionary of options that can be provided to the train method besides the train and test dataset """

        return self.__dict__

    def data_loader(self, temp_path, dataset = None ):

        from datasets.care_2d import load_data
        return load_data(temp_path)

    def train(self,train, test, datafraction = 1.):

        """setup the resnet and run the train function"""

        datafraction = float(datafraction)
        if datafraction > 1.0 or datafraction < 0:
            logging.error("resnet :: datafraction can only be [0,1]")

        #TODO: this if clause is non-sense, there must be a better way
        if "keras" == self.backend.lower():
            from .keras_details import care_denoise2d_details as keras_care_denoise2d
            return keras_care_denoise2d.train(train,test,datafraction,self.__dict__)

        if "tf.keras" == self.backend.lower() or "tensorflow.keras" == self.backend.lower():
            from .keras_details import tfkeras_care_denoise2d_details as tfkeras_care_denoise2d
            return tfkeras_care_denoise2d.train(train,test,datafraction,self.__dict__)

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
            elif self.backend.lower() == "tensorflow.keras" or self.backend.lower() == "tf.keras":
                import tensorflow as tf
                value = "tensorflow:{ver},tf.keras:{kver}".format(ver=tf.__version__,kver=tf.keras.__version__)
            else:
                value = "unknown:0.0"

        return value
