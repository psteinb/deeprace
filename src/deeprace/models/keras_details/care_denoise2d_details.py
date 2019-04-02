"""
 define a residual unet ("resunet") for image restoration


"""

from __future__ import print_function, unicode_literals, absolute_import, division
import logging
import math
import numpy as np
import os
import time
import importlib
from deeprace.models.tools.utils import versiontuple
from deeprace.models.keras_details.model_utils import to_disk


def can_train():
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    keras_found = importlib.util.find_spec('keras')

    available_backends = []

    if keras_found:
        from keras import __version__ as kv
        max_version = "2.2.4"
        min_version = "2.1.0"

        if versiontuple(kv,3) >= versiontuple(min_version,3) and versiontuple(kv,3) <= versiontuple(max_version,3):
            available_backends.append("keras")
        else:
            logging.debug("your keras version %s is not supported (%s - %s)",str(kv),min_version,max_version)

    return available_backends

def train(train, test, datafraction, optsdict):

    datafraction = float(datafraction)
    if datafraction > 1.0 or datafraction < 0:
        logging.error("resnet :: datafraction can only be [0,1]")
        return None

    from keras.layers import Input, Dropout, Activation, BatchNormalization
    from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
    from keras.layers.merge import Concatenate, Add
    from keras.models import Model
    from keras.utils import multi_gpu_model

    import keras.backend as K
    from keras.optimizers import Adam

    from deeprace.models.keras_details.callbacks import stopwatch
    from deeprace.models.keras_details.model_utils import model_size
    from deeprace.models.care_denoise import name

    batch_size=int(optsdict["batch_size"])
    epochs=int(optsdict["epochs"])
    if epochs <= 0:
        epochs = 60
        optsdict["epochs"] = epochs

    optsdict["n_gpus"] = int(optsdict["n_gpus"])
    depth = int(optsdict["depth"])
    model_type = name(optsdict["n_dims"],depth)

    fbase = int(optsdict["filter_base"])
    nrow = int(optsdict["n_row"])
    ncol = int(optsdict["n_col"])
    ncpd = int(optsdict["n_conv_per_depth"])



    if type(optsdict["checkpoint_epochs"]) == type(str()):
        checkpoint_epochs = bool(strtobool(optsdict["checkpoint_epochs"]))
    else:
        checkpoint_epochs = optsdict["checkpoint_epochs"]

    logging.info("received options: %s", optsdict)
    logging.info("%s (%i epochs):: batch_size = %i, depth = %i, checkpoint %i", model_type, epochs,batch_size,depth,checkpoint_epochs)

    nsamples_train = int(math.floor(train[0].shape[0]*datafraction))
    nsamples_train = int(math.floor(train[0].shape[0]*datafraction))

    x_train, y_train = train[0],train[-1]

    if datafraction != 1.0:
        x_train = train[0][:nsamples_train,]
        y_train = train[-1][:nsamples_train,]

    logging.info("using x=%s y=%s for training (original: x=%s y=%s) " % (x_train.shape, y_train.shape,train[0].shape, train[1].shape))

    # Input image dimensions.
    #input_shape = x_train.shape[1:]

    def conv_block2(n_filter, n1, n2,
                    activation="relu",
                    border_mode="same",
                    dropout=0.0,
                    batch_norm=False,
                    init="glorot_uniform",
                    **kwargs
                    ):
        def _func(lay):
            s = Conv2D(n_filter, (n1, n2), padding=border_mode,
                       kernel_initializer=init,
                       **kwargs)(lay)
            s = Activation(activation)(s)
            if batch_norm:
                s = BatchNormalization()(s)
            if dropout is not None:
                s = Dropout(dropout)(s)
            return s

        return _func





    def conv_sep_block2(n_filter, n1, n2,
                    activation="relu",
                    border_mode="same",
                    dropout=0.0,
                    batch_norm=False,
                    init="glorot_uniform",
                    **kwargs
                    ):
        def _func(lay):
            s = SeparableConv2D(n_filter, (n1, n2), padding=border_mode, kernel_initializer=init,**kwargs)(lay)
            if batch_norm:
                s = BatchNormalization()(s)
            s = Activation(activation)(s)
            if dropout is not None:
                s = Dropout(dropout)(s)
            return s

        return _func




    def unet_block(n_depth=2, n_filter_base=16, n_row=3, n_col=3, n_conv_per_depth=2,
                   activation="relu",
                   batch_norm=False,
                   dropout=0.0,
                   last_activation=None,
                   pool = (2,2),
                   weight_decay = None,
                   dilation_rate = (1,1),
                   prefix=''):
        """"""

        if last_activation is None:
            last_activation = activation

        if K.image_data_format() == "channels_last":
            channel_axis = -1
        else:
            channel_axis = 1

        def _name(s):
            return prefix+s

        if weight_decay is None:
            kernel_regularizer = None
        else:
            kernel_regularizer = regularizers.l2(weight_decay)

        def _func(input):
            skip_layers = []
            layer = input

            # down ...
            for n in range(n_depth):
                for i in range(n_conv_per_depth):
                    layer = conv_block2(n_filter_base * 2 ** n, n_row, n_col,
                                        dropout=dropout,
                                        activation=activation,
                                        batch_norm=batch_norm,
                                        dilation_rate  =dilation_rate,
                                        kernel_regularizer=kernel_regularizer,
                                        name = _name("down_level_%s_no_%s"%(n,i)))(layer)
                skip_layers.append(layer)
                layer = MaxPooling2D(pool, name = _name("max_%s"%n))(layer)


            # middle
            for i in range(n_conv_per_depth - 1):
                layer = conv_block2(n_filter_base * 2 ** n_depth, n_row, n_col,
                                    dropout=dropout,
                                    activation=activation,
                                    dilation_rate  =dilation_rate,
                                    kernel_regularizer=kernel_regularizer,
                                    batch_norm=batch_norm,name = _name("middle_%s"%i))(layer)

            layer = conv_block2(n_filter_base * 2 ** (n_depth - 1), n_row, n_col,
                                dropout=dropout,
                                activation=activation,
                                dilation_rate  =dilation_rate,
                                kernel_regularizer=kernel_regularizer,
                                batch_norm=batch_norm,name = _name("middle_%s"%n_conv_per_depth))(layer)

            # ...and up with skip layers
            for n in reversed(range(n_depth)):
                layer = Concatenate(axis = channel_axis)([UpSampling2D(pool)(layer), skip_layers[n]])
                for i in range(n_conv_per_depth - 1):
                    layer = conv_block2(n_filter_base * 2 ** n, n_row, n_col,
                                        dropout=dropout,
                                        activation=activation,
                                        dilation_rate  =dilation_rate,
                                        kernel_regularizer=kernel_regularizer,
                                        batch_norm=batch_norm,name = _name("up_level_%s_no_%s"%(n,i)))(layer)

                layer = conv_block2(n_filter_base * 2 ** max(0, n - 1), n_row, n_col,
                                    dropout=dropout,
                                    kernel_regularizer=kernel_regularizer,
                                    dilation_rate  =dilation_rate,
                                    activation=activation if n > 0 else last_activation,
                                    batch_norm=batch_norm, name = _name("up_level_%s_no_%s"%(n,n_conv_per_depth)))(layer)

            return layer

        return _func




    def resunet_model(input_shape,
                      last_activation,
                      n_depth=2,
                      n_filter_base=16,
                      n_row=3,
                      n_col=3,
                      n_conv_per_depth=2,
                      activation="relu",
                      batch_norm=False,
                      dropout=0.0,
                      weight_decay = None):
        if last_activation is None:
            raise ValueError("last activation has to be given (e.g. 'sigmoid'. 'relu')!")

        input = Input(input_shape, name="input")

        if K.image_data_format() == "channels_last":
            n_channels = input_shape[-1]
        else:
            n_channels = input_shape[0]


        unet = unet_block(n_depth, n_filter_base, n_row, n_col,
                          activation=activation, n_conv_per_depth=n_conv_per_depth,
                          dropout=dropout,
                          batch_norm = batch_norm,weight_decay = weight_decay)(
            input)

        final = Add()([Conv2D(n_channels, (1, 1), activation='linear')(unet), input])
        final = Activation(activation=last_activation)(final)

        return Model(inputs=input, outputs=final)

    model = None
    if optsdict["n_gpus"] != 1 and "tensorflow" in K.backend().lower():
        import tensorflow as tf
        with tf.device('/cpu:0'):
            temp_model = resunet_model(input_shape = (None,None,1),
                          last_activation = "linear",
                          n_depth=depth,
                          n_filter_base=fbase,
                          n_row=nrow,
                          n_col=ncol,
                          n_conv_per_depth=ncpd,
                          activation="relu",
                          )
        model = multi_gpu_model(temp_model, gpus=optsdict["n_gpus"])

    else:
        model = resunet_model(input_shape = (None,None,1),
                          last_activation = "linear",
                          n_depth=depth,
                          n_filter_base=fbase,
                          n_row=nrow,
                          n_col=ncol,
                          n_conv_per_depth=ncpd,
                          activation="relu",
                          )


    model.compile(optimizer=Adam(lr=0.0005), loss="mse",
                  metrics=['accuracy'# ,'top_k_categorical_accuracy'
                  ])

    if logging.getLogger().level == logging.DEBUG:
        model.summary()

    stopw = stopwatch()

    callbacks = [stopw]

    # Prepare model model saving directory.
    save_dir = os.path.join(optsdict["scratchspace"], 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    if checkpoint_epochs:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

            checkpoint = ModelCheckpoint(filepath=filepath,
                                         monitor='val_acc',
                                         verbose=1,
                                         save_best_only=True)
            callbacks.append(checkpoint)

    hist = model.fit(x_train,y_train,
                        epochs = epochs,
                        batch_size = batch_size,
                        validation_split = float(optsdict["validation_split"]),
                        shuffle = True,
                        callbacks=callbacks
                        )

    weights_fname = "{date}_{finishtime}_deeprace_{modeldescr}_finalweights.h5".format(date=time.strftime("%Y%m%d"),
                                                                                       finishtime=time.strftime("%H%M%S"),
                                                                                       modeldescr=model_type)

    to_disk(model,
            os.path.join(optsdict["scratchspace"],weights_fname))

    return hist.history, stopw, { 'num_weights' : model_size(model) }

