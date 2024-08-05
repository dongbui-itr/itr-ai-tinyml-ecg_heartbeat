"""MobileNet v2 models for Keras.

# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""



from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, Conv1D, GlobalAveragePooling1D
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D, ReLU, DepthwiseConv1D, Flatten
from keras import backend as K

import tensorflow as tf


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv1d_net(x,
               filters,
               kernel_size,
               strides=1,
               padding='same',
               act=True,
               bn=True,
               rate=0.5,
               name=""):

    if bn:
        x = BatchNormalization(axis=-1)(x)

    if act:
        x = ReLU(max_value=6.0)(x)

    if rate < 1.0:
        x = Dropout(rate=rate)(x)

    x = Conv1D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding)(x)

    return x


def _conv_block_1D(inputs,
                   filters,
                   kernel,
                   strides,
                   padding='same',
                   act=True,
                   bn=True,
                   name=""):

    return conv1d_net(inputs,
                   filters,
                   kernel,
                   act=act,
                   bn=bn,
                   strides=strides,
                   padding=padding,
                   name=name)


def _bottleneck_1D(inputs, filters, kernel, t, alpha, s, r=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block_1D(inputs, tchannel, 1, 1)

    x = DepthwiseConv1D(kernel, strides=s, depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv1D(cchannel, 1, strides=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x

def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)


def _inverted_residual_block_1D(inputs, filters, kernel, t, alpha, strides, n):
    x = _bottleneck_1D(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = _bottleneck_1D(x, filters, kernel, t, alpha, 1, True)

    return x

def MobileNetv2_1D(input_shape,
                   num_of_class,
                   output_shape,
                   alpha=1.0,
                   rate=0.5,
                   name='MobileNetv2_1D'):
    inputs = Input(shape=(input_shape, 1))

    first_filters = _make_divisible(32 * alpha, 8)
    x = _conv_block_1D(inputs, first_filters, 3, strides=2)

    x = _inverted_residual_block_1D(x, 16, 3, t=1, alpha=alpha, strides=1, n=1)
    x = _inverted_residual_block_1D(x, 24, 3, t=6, alpha=alpha, strides=2, n=2)
    x = _inverted_residual_block_1D(x, 24, 3, t=6, alpha=alpha, strides=1, n=2)
    x = _inverted_residual_block_1D(x, 32, 3, t=6, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block_1D(x, 64, 3, t=6, alpha=alpha, strides=2, n=4)
    x = _inverted_residual_block_1D(x, 96, 3, t=6, alpha=alpha, strides=1, n=3)
    x = _inverted_residual_block_1D(x, 160, 3, t=6, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block_1D(x, 320, 3, t=6, alpha=alpha, strides=1, n=1)

    if alpha > 1.0:
        last_filters = _make_divisible(output_shape * alpha, 8)
    else:
        last_filters = output_shape

    x = _conv_block_1D(x, last_filters, 3, strides=1)
    # x = GlobalAveragePooling1D()(x)
    x = Reshape((last_filters, -1))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv1D(x.shape[-1], 3, 1, padding='same')(x)

    # output = Activation('softmax', name='softmax')(x)

    logits_layer1 = tf.keras.layers.Dense(num_of_class)(x)
    lstm_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer2 = tf.keras.layers.Dense(num_of_class)(lstm_layer)
    logits_layer = tf.keras.layers.Add()([logits_layer1, logits_layer2])

    output = Activation('softmax', name='softmax')(logits_layer)

    model = Model(inputs, output, name=name)

    # model.summary()

    return model


if __name__ == '__main__':
    model = MobileNetv2_1D(2500, 5, 100, 1.0)
    print(model.summary())
