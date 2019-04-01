import re
import numpy as np
import os
import logging
import math
import time

from distutils.util import strtobool
from models.base import base_model


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
        self.weights_file = None
        self.backend = "keras"
        self.n_gpus = 1
        if self.available_datasets() and len(self.available_datasets())>0:
            self.dataset = self.available_datasets()[0]
        else:
            self.dataset = None

    def available_datasets(self):
        datasets = []

        from models.keras_details import care_denoise2d_details as keras_net
        if keras_net.can_train() != []:
            if not 'cifar10' in datasets:
                datasets.append('cifar10')

        from models.keras_details import tfkeras_care_denoise2d_details as tfkeras_net
        if tfkeras_net.can_train() != []:
            if not 'cifar10' in datasets:
                datasets.append('cifar10')

        return datasets

    def provides(self):
        """ provide a tuple,
        item[0] yields a list which models can be provided by this module
        item[1] yields a list which backends can be used
        item[2] yields a list which datasets can used
        """

        # models
        possible_values = [3,5,7,9,18,27]

        value = [ name(n=i,version=1) for i in possible_values ]

        possible_values.append(111)
        value.extend( [ name(n=i,version=2) for i in possible_values ] )

        #TODO: automate this
        backends = []
        datasets = self.available_datasets()

        from .keras_details import resnet_details as keras_resnet
        if keras_resnet.can_train() != []:
            backends.extend(keras_resnet.can_train())

        from .keras_details import tfkeras_resnet_details as tfkeras_resnet
        if tfkeras_resnet.can_train() != []:
            backends.extend(tfkeras_resnet.can_train())

        from .tf_details import resnet_details as tf_resnet
        if tf_resnet.can_train() != []:
            backends.extend(tf_resnet.can_train())

        ### datasets

        return value, backends, datasets

    def options(self):
        """ return a dictionary of options that can be provided to the train method besides the train and test dataset """

        return self.__dict__

    def data_loader(self, temp_path, dataset_name = "cifar10" ):

        if 'cifar10' not in dataset_name:
            logging.error("resnet is unable to load unknown dataset %s (must be %s)",dataset_name,",".join(self.available_datasets()))
            return None

        #TODO: this if clause is non-sense, there must be a better way
        if "keras" == self.backend.lower():
            from .keras_details.resnet_details import data_loader
            return data_loader(temp_path, dataset_name)

        #TODO: enable pure tensorflow again, once TF2 has matured
        # elif "tf" == self.backend.lower() or "tensorflow" == self.backend.lower():
        #     from .tf_details.resnet_details import data_loader
        #     return data_loader(temp_path, dataset_name)

        elif ("tf" in self.backend.lower() or "tensorflow" in self.backend.lower()) and "keras" in self.backend.lower():
            from .keras_details.tfkeras_resnet_details import data_loader
            return data_loader(temp_path, dataset_name)


    def train(self,train, test, datafraction = 1.):

        """setup the resnet and run the train function"""

        datafraction = float(datafraction)
        if datafraction > 1.0 or datafraction < 0:
            logging.error("resnet :: datafraction can only be [0,1]")

        #TODO: this if clause is non-sense, there must be a better way
        if "keras" == self.backend.lower():
            from .keras_details import resnet_details as keras_resnet
            logging.info("using keras backend")
            return keras_resnet.train(train,test,datafraction,self.__dict__)
        if "tf" == self.backend.lower() or "tensorflow" == self.backend.lower():
            from .tf_details import resnet_details as tf_resnet
            logging.info("using tensorflow backend")
            return tf_resnet.train(train,test,datafraction,self.__dict__)

        if "tf.keras" == self.backend.lower() or "tensorflow.keras" == self.backend.lower():
            from .keras_details import tfkeras_resnet_details as tfkeras_resnet
            logging.info("using tensorflow.keras backend")
            return tfkeras_resnet.train(train,test,datafraction,self.__dict__)


    def infer(self, data ,  num_inferences = 1):

        """setup the resnet and run the train function"""

        if "keras" == self.backend.lower():
            from .keras_details import resnet_details as keras_resnet
            logging.info("using keras backend")
            return keras_resnet.infer(data, num_inferences ,self.__dict__)

        if "tf.keras" == self.backend.lower() or "tensorflow.keras" == self.backend.lower():
            from .keras_details import tfkeras_resnet_details as tfkeras_resnet
            logging.info("using tensorflow.keras backend")
            return tfkeras_resnet.infer(data, num_inferences ,self.__dict__)
        else:
            logging.warning("tensorflow backend not implemented yet")
            return None, None, None

        # if "tf" == self.backend.lower() or "tensorflow" == self.backend.lower():
        #     from .tf_details import resnet_details as tf_resnet
        #     logging.info("using tensorflow")
        #     return tf_resnet.train(train,test,datafraction,self.__dict__)

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
