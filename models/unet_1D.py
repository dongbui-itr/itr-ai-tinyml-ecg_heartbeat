from keras.layers import Conv1D, Input, Conv1DTranspose, Flatten, Dense
from keras.layers import MaxPooling1D, BatchNormalization, UpSampling1D, ZeroPadding1D
from keras.layers import concatenate, add, ZeroPadding1D, LeakyReLU
from keras.models import Model
import tensorflow as tf
import numpy as np


def FCNN(input_shape=None, conv_activation='LeakyReLU'):
    encoder_inputs = inputs = Input(shape=(input_shape, 1))
    x_in_1 = Conv1D(filters=40, kernel_size=16, strides=2, padding="same", activation=conv_activation)(encoder_inputs)
    x_in_1 = Conv1D(filters=20, kernel_size=16, strides=2, padding="same", activation=conv_activation)(x_in_1)
    x_in_1 = Conv1D(filters=20, kernel_size=16, strides=2, padding="same", activation=conv_activation)(x_in_1)
    x_in_1 = Conv1D(filters=20, kernel_size=16, strides=2, padding="same", activation=conv_activation)(x_in_1)
    x_in_1 = Conv1D(filters=40, kernel_size=16, strides=2, padding="same", activation=conv_activation)(x_in_1)
    x_in_1 = Conv1D(filters=1, kernel_size=16, strides=1, padding="same", activation=conv_activation)(x_in_1)
    x_out_1 = Conv1DTranspose(1, kernel_size=16, strides=1, padding="same", activation=conv_activation)(x_in_1)
    x_out_1 = Conv1DTranspose(40, kernel_size=16, strides=2, padding="same", activation=conv_activation)(x_out_1)
    x_out_1 = Conv1DTranspose(20, kernel_size=16, strides=2, padding="same", activation=conv_activation)(x_out_1)
    x_out_1 = Conv1DTranspose(20, kernel_size=16, strides=2, padding="same", activation=conv_activation)(x_out_1)
    x_out_1 = Conv1DTranspose(20, kernel_size=16, strides=2, padding="same", activation=conv_activation)(x_out_1)
    x_out_1 = Conv1DTranspose(40, kernel_size=16, strides=2, padding="same", activation=conv_activation)(x_out_1)
    x_out_1 = Conv1DTranspose(filters=1, kernel_size=16, strides=1, padding="same", activation=conv_activation)(x_out_1)
    # x_out_1=Reshape((-1,1,size_output))(x_out_1)
    x_out_1 = Flatten()(x_out_1)
    # x_out_1 = Conv1D(128, 3, strides=1, padding='same')(x_out_1)
    # x_out_1 = Dense(size_output, activation='tanh')(x_out_1)
    out = Dense(input_shape, activation='tanh')(x_out_1)

    autoencoder = Model(inputs=encoder_inputs, outputs=out)

    autoencoder.summary()

    # autoencoder.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=5e-4), loss='mse')
    return autoencoder


def AE_1d(input_shape=None,
          conv_activation='LeakyReLU'):
    inputs = Input(shape=(input_shape, 1))
    inputs_pad = ZeroPadding1D(padding=(15, 1))(inputs)
    x1 = Conv1D(64, 4, 2, padding='same', activation=conv_activation)(inputs_pad)
    x2 = Conv1D(32, 4, 2, padding='same', activation=conv_activation)(x1)
    x3 = Conv1D(16, 4, 2, padding='same', activation=conv_activation)(x2)
    x4 = Conv1D(1, 1, 1, activation=conv_activation)(x3)
    x5 = Conv1DTranspose(1, 1, 1, activation=conv_activation)(x4)
    x6 = Conv1DTranspose(16, 4, 2, activation=conv_activation)(x5)
    x7 = Conv1DTranspose(32, 4, 2, activation=conv_activation)(x6)
    x8 = Conv1DTranspose(64, 4, 2, activation=conv_activation)(x7)
    x9 = Conv1D(2, 4, 2)(x8)
    x10 = Flatten()(x9)
    output = x10[:, :2500]
    # with tf.compat.v1.variable_scope('collected') as scope:
    #     bbbbbx = x10[:, :2500]
    #     bbbbxx = x10[:, 4:2504]
    #     bbbxxx = x10[:, 8:2508]
    #     bbxxxx = x10[:, 12:2512]
    #     bxxxxx = x10[:, 16:]

    # output = add((bbbbbx, bbbbxx, bbbxxx, bbxxxx, bxxxxx), name='add')

    model = Model(inputs, output)
    # model.summary()

    return model


def fire_module(x, fire_id, squeeze=16, expand=64, conv_activation='linear'):
    f_name = "fire{0}_{1}"

    f_name_1 = f_name.format(fire_id, "squeeze1")
    x = Conv1D(squeeze, 1, activation=conv_activation, padding='same',
               name=f_name_1)(x)
    x = BatchNormalization(axis=1)(x)

    left = Conv1D(expand, 1, activation=conv_activation, padding='same',
                  name=f_name.format(fire_id, "expand1"))(x)
    right = Conv1D(expand, 3, activation=conv_activation, padding='same',
                   name=f_name.format(fire_id, "expand3"))(x)
    f_name_2 = f_name.format(fire_id, "concat")
    x = concatenate([left, right], axis=2, name=f_name_2)
    return x


def SqueezeUNet(feature_len,
                num_of_class=3,
                model_width=16,
                deconv_ksize=2,
                conv_activation='relu',
                name='SqueezeUNet'):

    inputs = Input(shape=(feature_len, 1))
    # inputs_padding = ZeroPadding1D(6)(inputs)
    """SqueezeUNet is a implementation based in SqueezeNetv1.1 and unet for semantic segmentation

    """
    # CB1
    x01 = Conv1D(model_width, 3, strides=2, padding='same', activation=conv_activation, name='conv1')(inputs)

    # CB2
    x02 = MaxPooling1D(pool_size=2, strides=2, name='pool1', padding='same')(x01)  # maxpooling 2 1

    # CB3
    x03 = fire_module(x02, fire_id=1, squeeze=model_width // 4, expand=model_width, conv_activation=conv_activation)
    x04 = fire_module(x03, fire_id=2, squeeze=model_width // 4, expand=model_width, conv_activation=conv_activation)
    x05 = MaxPooling1D(pool_size=2, strides=2, name='pool3', padding="same")(x04)

    # CB4
    x06 = fire_module(x05, fire_id=3, squeeze=(model_width * 2) // 4, expand=model_width * 2, conv_activation=conv_activation)
    x07 = fire_module(x06, fire_id=4, squeeze=(model_width * 2) // 4, expand=model_width * 2, conv_activation=conv_activation)
    x08 = MaxPooling1D(pool_size=2, strides=2, name='pool5', padding="same")(x07)

    # CB5
    x09 = fire_module(x08, fire_id=5, squeeze=(model_width * 3) // 4, expand=model_width * 3, conv_activation=conv_activation)
    x10 = fire_module(x09, fire_id=6, squeeze=(model_width * 3) // 4, expand=model_width * 3, conv_activation=conv_activation)
    x11 = fire_module(x10, fire_id=7, squeeze=(model_width * 4) // 4, expand=model_width * 4, conv_activation=conv_activation)
    x12 = fire_module(x11, fire_id=8, squeeze=(model_width * 4) // 4, expand=model_width * 4, conv_activation=conv_activation)

    up1 = concatenate([Conv1DTranspose(model_width * 3, deconv_ksize, strides=1, padding='same', activation=conv_activation)(x12), x10], axis=2)

    up2 = fire_module(up1, fire_id=9, squeeze=(model_width * 3) // 4, expand=model_width * 3, conv_activation=conv_activation)

    up3 = concatenate([Conv1DTranspose(model_width * 2, deconv_ksize, strides=1, padding='same', activation=conv_activation)(up2), x08], axis=2)
    up4 = fire_module(up3, fire_id=10, squeeze=(model_width * 2) // 4, expand=model_width, conv_activation=conv_activation)

    # EB1
    up5 = concatenate([Conv1DTranspose(model_width, deconv_ksize, strides=2, padding='same', activation=conv_activation)(up4), x05], axis=2)

    up6 = fire_module(up5, fire_id=11, squeeze=model_width // 4, expand=model_width, conv_activation=conv_activation)

    # EB2
    up7 = concatenate([Conv1DTranspose(model_width // 2, deconv_ksize, strides=2, padding='same', activation=conv_activation)(up6), x02], axis=2)
    up8 = fire_module(up7, fire_id=12, squeeze=model_width // 4, expand=model_width // 2, conv_activation=conv_activation)
    up9 = UpSampling1D(size=2)(up8)

    x = concatenate([up9, x01], axis=2)
    x = Conv1D(model_width, 3, strides=1, padding='same', activation='relu')(x)
    x = UpSampling1D(size=2)(x)

    x = Conv1D(8, 5, strides=2, padding="same", activation=conv_activation, name="last_cnn_1")(x)
    x = Conv1D(8, 5, strides=2, padding="same", activation=conv_activation, name="last_cnn_2")(x)
    x = Conv1D(8, 5, strides=2, padding="same", activation=conv_activation, name="last_cnn_3")(x)
    x = Conv1D(8, 5, strides=2, padding="same", activation=conv_activation, name="last_cnn_4")(x)

    output = Conv1D(num_of_class, 5, strides=2, padding="same", activation=conv_activation, name="output")(x)

    # with tf.compat.v1.variable_scope('collected') as scope:
    #     bbbbbx = x[:, :2500]
    #     bbbbxx = x[:, 3:2503]
    #     bbbxxx = x[:, 6:2506]
    #     bbxxxx = x[:, 10:2510]

    # output = add((bbbbbx, bbbbxx, bbbxxx, bbxxxx), name='add')

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model


def test_model():
    # FCNN(input_shape=2500)
    # AE_1d(input_shape=2500)
    SqueezeUNet(feature_len=640)


if __name__ == '__main__':
    test_model()
