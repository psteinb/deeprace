"""
 define a residual unet ("resunet") for image restoration


"""

from __future__ import print_function, unicode_literals, absolute_import, division

from keras.layers import Input, Dropout, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.merge import Concatenate, Add
from keras.models import Model
import keras.backend as K

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


from care_denoise2d_data import create_data

if __name__ == '__main__':


    X,Y = create_data()




    model = resunet_model(input_shape = (None,None,1),
                          last_activation = "linear",
                          n_depth=2,
                          n_filter_base=32,
                          n_row=3,
                          n_col=3,
                          n_conv_per_depth=2,
                          activation="relu",
                          )


    model.compile(optimizer=Adam(lr=0.0005), loss="mse")

    history = model.fit(X,Y,
                        epochs = 60,
                        batch_size = 16,
                        validation_split = 0.1,
                        shuffle = True
                        )
