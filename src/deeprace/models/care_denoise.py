import re
import numpy as np
import os
import logging
import math
import time
import glob

from distutils.util import strtobool
from deeprace.base import base_model


def name(ndims=2, ndepth=2):
    """ encrypt n and version into a standardized string """

    # Model name, depth and version
    value = 'care_denoise_%dDdepth%d' % (ndims, ndepth)

    return value


def params_from_name(name):
    """ function that extracts a dictionary of parameters from a given name,
    e.g. care_denoise2d_depth2 would result in { 'n_depth' : 2 },
    this is the inverse of the 'name' function
    """

    found = re.findall(r'\d+', name)
    value = {'n_depth': None}
    if not len(found) == 2:
        value['n_depth'] = 2
        value['n_dims'] = 2
    else:
        value['n_dims'] = int(found[0])
        value['n_depth'] = int(found[-1])

    return value


class model(base_model):

    def __init__(self):

        self.depth = 2  # depth
        self.n_dims = 2

        self.version = 1
        self.filter_base = 16
        self.n_row = 3
        self.n_col = 3
        self.n_conv_per_depth = 2

        self.batch_size = 32
        self.epochs = 60
        self.checkpoint_epochs = False
        self.scratchspace = os.getcwd()
        self.backend = "keras"
        self.n_gpus = 1
        self.validation_split = 0.1
        self.dataset = self.available_datasets()[0]

    def available_datasets(self):
        datasets = []

        from deeprace.models.keras_details import care_denoise2d_details as keras_net
        if keras_net.can_train() != []:
            if not 'care_2d' in datasets:
                datasets.append('care_2d')

        from deeprace.models.keras_details import tfkeras_care_denoise2d_details as tfkeras_net
        if tfkeras_net.can_train() != []:
            if not 'care_2d' in datasets:
                datasets.append('care_2d')

        return datasets

    def provides(self):
        """ provide a tuple,
        item[0] yields a list which models can be provided by this module
        item[1] yields a list which backends can be used
        item[2] yields a list which datasets can used
        """

        possible_values = {"2D": [2]}

        value = [name(ndims=2, ndepth=i) for i in possible_values["2D"]]

        # TODO: automate this
        backends = []
        datasets = self.available_datasets()

        from deeprace.models.keras_details import care_denoise2d_details as keras_net
        if keras_net.can_train() != []:
            backends.extend(keras_net.can_train())

        from deeprace.models.keras_details import tfkeras_care_denoise2d_details as tfkeras_net
        if tfkeras_net.can_train() != []:
            backends.extend(tfkeras_net.can_train())

        return value, backends, datasets

    def options(self):
        """ return a dictionary of options that can be provided to the train method besides the train and test dataset """

        return self.__dict__

    def data_loader(self, temp_path, dataset_name='care_2d'):

        if 'care_2d' not in dataset_name:
            logging.error("care_denoise is unable to load unknown dataset %s (must be %s)",
                          dataset_name, ",".join(self.available_datasets()))
            return None

        from datasets.care_2d import load_data
        train = load_data(temp_path)
        ntrain = train[0].shape[0]
        logging.debug("[care_denoise::data loader] training dataset x=%s y=%s ", train[0].shape, train[-1].shape)

        return (train, None, ntrain, 0)

    def train(self, train, test, datafraction=1.):
        """setup the resnet and run the train function"""

        datafraction = float(datafraction)
        if datafraction > 1.0 or datafraction < 0:
            logging.error("resnet :: datafraction can only be [0,1]")
            return None

        # TODO: this if clause is non-sense, there must be a better way
        if "keras" == self.backend.lower():
            from deeprace.models.keras_details import care_denoise2d_details as keras_care_denoise2d
            logging.info("using keras")
            return keras_care_denoise2d.train(train, test, datafraction, self.__dict__)

        if "tf.keras" == self.backend.lower() or "tensorflow.keras" == self.backend.lower():
            from deeprace.models.keras_details import tfkeras_care_denoise2d_details as tfkeras_care_denoise2d
            logging.info("using tensorflow.keras")
            return tfkeras_care_denoise2d.train(train, test, datafraction, self.__dict__)

    def infer(self, data, num_inferences=1):
        """setup the resnet and run the train function"""

        logging.warning("inference not implemented yet for care_denoise")
        return None, None, None

        # if "keras" == self.backend.lower():
        #     from deeprace.models.keras_details import resnet_details as keras_resnet
        #     logging.info("using keras backend")
        #     return keras_resnet.infer(data, num_inferences ,self.__dict__)

        # if "tf.keras" == self.backend.lower() or "tensorflow.keras" == self.backend.lower():
        #     from deeprace.models.keras_details import tfkeras_resnet_details as tfkeras_resnet
        #     logging.info("using tensorflow.keras backend")
        #     return tfkeras_resnet.infer(data, num_inferences ,self.__dict__)

        # if "tf" == self.backend.lower() or "tensorflow" == self.backend.lower():
        #     from deeprace.models.tf_details import resnet_details as tf_resnet
        #     logging.info("using tensorflow")
        #     return tf_resnet.train(train,test,datafraction,self.__dict__)

    def versions(self):

        value = ""

        if self.backend.lower().startswith("keras"):

            import keras
            from keras import backend as K

            value = "keras:{kver},backend:{bname}".format(kver=keras.__version__, bname=K.backend())

            if K.tf:
                value += ":" + K.tf.__version__
            else:
                # the following is untested!
                try:
                    if K.th:
                        value += ":" + K.th.__version__
                    else:
                        if K.cntk:
                            value += ":" + K.cntk.__version__
                except BaseException:
                    value += ":???"

        else:

            if self.backend.lower() == "tensorflow" or self.backend.lower() == "tf":
                import tensorflow as tf
                value = "tensorflow:{ver}".format(ver=tf.__version__)
            elif self.backend.lower() == "tensorflow.keras" or self.backend.lower() == "tf.keras":
                import tensorflow as tf
                value = "tensorflow:{ver},tf.keras:{kver}".format(ver=tf.__version__, kver=tf.keras.__version__)
            else:
                value = "unknown:0.0"

        return value
