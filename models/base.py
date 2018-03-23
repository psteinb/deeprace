#!/usr/bin/env python3
import abc


class base_model(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def provides():
        """ provide a list of strings which denote which models can be provided by this module """
        pass

    @abc.abstractmethod
    def options():
        """ return a dictionary of options that can be provided to the train method besides the train and test dataset """
        pass

    @abc.abstractmethod
    def data_loader(temp_path="."):
        """ load input data into memory, return a tuple train <np.narray>, test <np.narray>, ntrain, ntest """
        pass


    @abc.abstractmethod
    def train(train, test, datafraction = 1.):
        """ run the training of the model with train data <train|np.array> and <test|np.array>, but use only datafraction for training """
        pass
