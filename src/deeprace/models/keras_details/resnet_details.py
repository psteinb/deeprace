import logging
import math
import numpy as np
import os
import time
import importlib
import datetime
import random

from deeprace.models.tools.utils import versiontuple
from deeprace.models.keras_details.model_utils import to_disk

def data_loader(temp_path, dsname = "cifar10" ):

    if "cifar100" in dsname.lower():
        logging.error("resnet :: cifar100 is not supported yet")
        sys.exit(1)
        #TODO likely to yield an error
        # from deeprace.datasets import cifar100
        # train, test = cifar100.load_data(temp_path)
        # ntrain, ntest = train[0].shape[0], test[0].shape[0]
        # return train, test, ntrain, ntest
    elif "cifar10" in dsname.lower():
        from deeprace.datasets import cifar10
        train, test = cifar10.load_data(temp_path)
        ntrain, ntest = train[0].shape[0], test[0].shape[0]
        return train, test, ntrain, ntest



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


def compute_depth(n=3,version=1):
    value = 0
    if version == 1:
        value = n * 6 + 2
    elif version == 2:
        value = n * 9 + 2
    return value

def infer(data, num_inferences, optsdict):

    """ perform <num_inferences> on the given data """

    import keras
    from keras.models import model_from_json
    from keras.optimizers import Adam
    ####################################################################################################################
    ## LOADING THE MODEL
    ##
    no_ext = os.path.splitext(optsdict["weights_file"])[0]
    model_json = no_ext + '.json'

    if not os.path.exists(model_json):
        logging.error("%s does not exist, unable to load a model and thus perform inference!",model_json)
        return None, None, None

    model_weights = no_ext + '.h5'
    if not os.path.exists(model_weights):
        logging.error("%s does not exist, unable to load the model weights and thus perform inference!",model_json)
        return None, None, None

    ## Watch out for
    with open(model_json,"r") as f:
        json_str = f.read()
        f.close()

    model = keras.models.model_from_json(json_str)

    # Weights
    model.load_weights(model_weights)
    model.compile(optimizer=Adam())

    nsamples_infer = int(num_inferences) if int(num_inferences) <= data[0].shape[0] else data[0].shape[0]
    batch_size=int(optsdict["batch_size"]) if int(optsdict["batch_size"]) < nsamples_infer else 1

    n_iterations = nsamples_infer // batch_size
    timings = [0]*n_iterations
    predictions = [0.]*n_iterations

    for t in range(n_iterations):

        random_id = int(round(random.uniform(0,data[0].shape[0] - batch_size)))
        input_data = data[0][random_id:random_id+batch_size,...]

        start = datetime.datetime.now()
        result = model.evaluate(input_data,verbose=True, batch_size = batch_size)
        end = datetime.datetime.now()

        timings[t] = ((end-start).total_seconds())
        predictions[t] = result


    return predictions, timings, None

def train(train, test, datafraction, optsdict):

    """setup the resnet and run the train function"""

    datafraction = float(datafraction)
    if datafraction > 1.0 or datafraction < 0:
        logging.error("resnet :: datafraction can only be [0,1]")
        return None

    import keras
    from keras.layers import Dense, Conv2D, BatchNormalization, Activation
    from keras.layers import AveragePooling2D, Input, Flatten
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler
    from keras.callbacks import ReduceLROnPlateau
    from keras.preprocessing.image import ImageDataGenerator
    from keras.regularizers import l2
    from keras import backend as K
    from keras.models import Model
    from deeprace.models.keras_details.callbacks import stopwatch
    from deeprace.models.keras_details.model_utils import model_size, to_disk
    from keras.utils import multi_gpu_model

    batch_size=int(optsdict["batch_size"])
    epochs=int(optsdict["epochs"])
    if epochs <= 0:
        epochs = 200
        optsdict["epochs"] = epochs

    optsdict["n_gpus"] = int(optsdict["n_gpus"])
    depth = compute_depth(optsdict["n"],optsdict["version"])
    model_type = 'ResNet%dv%d' % (depth, optsdict["version"])

    if type(optsdict["data_augmentation"]) == type(str()):
        data_augmentation = bool(strtobool(optsdict["data_augmentation"]))
    else:
        data_augmentation = optsdict["data_augmentation"]

    if type(optsdict["checkpoint_epochs"]) == type(str()):
        checkpoint_epochs = bool(strtobool(optsdict["checkpoint_epochs"]))
    else:
        checkpoint_epochs = optsdict["checkpoint_epochs"]

    if type(optsdict["subtract_pixel_mean"]) == type(str()):
        subtract_pixel_mean = bool(strtobool(optsdict["subtract_pixel_mean"]))
    else:
        subtract_pixel_mean = optsdict["subtract_pixel_mean"]

    logging.info("received options: %s", optsdict)
    logging.info("%s (%i epochs):: batch_size = %i, depth = %i, data_augmentation/checkpoint/subtract_pixel_mean %i/%i/%i", model_type, epochs,batch_size,depth,data_augmentation,checkpoint_epochs,subtract_pixel_mean)

    nsamples_train = int(math.floor(train[0].shape[0]*datafraction))
    nsamples_test = int(math.floor(test[0].shape[0]*datafraction))

    x_train = train[0][:nsamples_train,...]
    y_train = train[-1][:nsamples_train,...]

    x_test = test[0][:nsamples_test,...]
    y_test = test[-1][:nsamples_test,...]

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    logging.info('x_train shape: %s, %i samples', str(x_train.shape),x_train.shape[0])
    logging.info('y_train shape: %s, %i samples', str(y_train.shape),y_train.shape[0])
    logging.info('x_test shape: %s, %i samples', str(x_test.shape),x_test.shape[0])
    logging.info('y_test shape: %s, %i samples', str(y_test.shape),y_test.shape[0])

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, optsdict["num_classes"])
    y_test = keras.utils.to_categorical(y_test, optsdict["num_classes"])


    def lr_schedule(epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
            print('Learning rate: ', lr)
        return lr


    def resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     batch_normalization=True,
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                activation-bn-conv (False)

        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
                x = conv(x)
        return x


    def resnet_v1(input_shape, depth, num_classes=10):
        """ResNet Version 1 Model builder [a]

        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)


        inputs = Input(shape=input_shape)
        x = resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     strides=strides)
                y = resnet_layer(inputs=y,
                                     num_filters=num_filters,
                                     activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model


    def resnet_v2(input_shape, depth, num_classes=10):
        """ResNet Version 2 Model builder [b]

        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = Input(shape=input_shape)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = resnet_layer(inputs=inputs,
                         num_filters=num_filters_in,
                         conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2    # downsample

                # bottleneck residual unit
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters_in,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=activation,
                                 batch_normalization=batch_normalization,
                                 conv_first=False)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters_in,
                                 conv_first=False)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 conv_first=False)
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters_out,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    generate_it = None
    if optsdict["version"] == 2:
        generate_it = resnet_v2 #(input_shape=input_shape, depth=depth)
    else:
        generate_it = resnet_v1 #(input_shape=input_shape, depth=depth)

    model = None
    if optsdict["n_gpus"] != 1 and "tensorflow" in K.backend().lower():
        import tensorflow as tf
        with tf.device('/cpu:0'):
            temp_model = generate_it(input_shape=input_shape, depth=depth)
        model = multi_gpu_model(temp_model, gpus=optsdict["n_gpus"])

    else:
        model = generate_it(input_shape=input_shape, depth=depth)


    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy','top_k_categorical_accuracy'])

    if logging.getLogger().level == logging.DEBUG:
        model.summary()


    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    stopw = stopwatch()

    callbacks = [lr_reducer, lr_scheduler, stopw]

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

    hist = None
    # Run training, with or without data augmentation.
    if not data_augmentation:
        logging.info('Not using data augmentation.')
        hist = model.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(x_test, y_test),
                         shuffle=True,
                         callbacks=callbacks)
    else:
        logging.info('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False
                         )

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                   steps_per_epoch=math.ceil(len(x_train) / batch_size),##required with keras 2.2.4
                                   validation_data=(x_test, y_test),
                                   epochs=epochs, verbose=1, workers=4,
                                   callbacks=callbacks)

    weights_fname = "{date}_{finishtime}_deeprace_{modeldescr}_finalweights.h5".format(date=time.strftime("%Y%m%d"),
                                                                                       finishtime=time.strftime("%H%M%S"),
                                                                                       modeldescr=model_type)

    to_disk(model,os.path.join(optsdict["scratchspace"],weights_fname))

    return hist.history, stopw, { 'num_weights' : model_size(model) }

