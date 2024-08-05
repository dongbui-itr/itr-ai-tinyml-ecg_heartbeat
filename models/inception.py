
#https://github.com/hfawaz/InceptionTime

from keras.layers import Conv1D, Concatenate, BatchNormalization, Activation, MaxPool1D, Add, Input, GlobalAveragePooling1D, Dense
from keras.models import Model

def _inception_module(input_tensor,
                      kernel_size,
                      nb_filters,
                      bottleneck_size,
                      stride = 1,
                      activation = 'linear',
                      use_bottleneck=True
                      ) :

    if use_bottleneck and int(input_tensor.shape[-1]) > 1 :
        input_inception = Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else :
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)) :
        conv_list.append(Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                             strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = Conv1D(filters=nb_filters, kernel_size=1,
                                 padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = Concatenate(axis=2)(conv_list)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    return x

def _shortcut_layer(self, input_tensor, out_tensor) :
    shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                     padding='same', use_bias=False)(input_tensor)
    # shortcut_y = normalization.BatchNormalization()(shortcut_y)
    shortcut_y = BatchNormalization()(shortcut_y)

    x = Add()([shortcut_y, out_tensor])
    x = Activation('relu')(x)
    return x

def inception_1d(input_shape, nb_classes, depth, use_residual=True) :
    input_layer = Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth) :
        x = _inception_module(x)
        if use_residual and d % 3 == 2 :
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = GlobalAveragePooling1D()(x)
    output_layer = Dense(nb_classes, activation='softmax')(gap_layer)

    return Model(inputs=input_layer, outputs=output_layer)