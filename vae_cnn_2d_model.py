# from keras.layers import Conv2D, Input, Conv2DTranspose, Flatten, Dense
# from keras.layers import MaxPooling2D, BatchNormalization, UpSampling2D
# from keras.layers import concatenate, add, ZeroPadding2D, LeakyReLU
# from keras.models import Model
import tensorflow as tf

import keras
layers = keras.layers
(Conv2D, Input, Conv2DTranspose, Flatten, Dense) = (layers.Conv2D, layers.Input, layers.Conv2DTranspose, layers.Flatten, layers.Dense)
(MaxPooling2D, BatchNormalization, UpSampling2D) = (layers.MaxPooling2D, layers.BatchNormalization, layers.UpSampling2D)
(concatenate, add, ZeroPadding2D, LeakyReLU, ReLU) = (layers.concatenate, layers.add, layers.ZeroPadding2D, layers.LeakyReLU, layers.ReLU)
Model = keras.models.Model


def fire_module(x, fire_id, squeeze=16, expand=64, conv_activation='linear'):
    f_name = "fire{0}_{1}"

    x = Conv2D(squeeze, 1, activation=conv_activation, padding='same',
               name=f_name.format(fire_id, "squeeze1"))(x)
    x = BatchNormalization(axis=1)(x)

    left = Conv2D(expand, 1, activation=conv_activation, padding='same',
                  name=f_name.format(fire_id, "expand1"))(x)
    right = Conv2D(expand, 3, activation=conv_activation, padding='same',
                   name=f_name.format(fire_id, "expand3"))(x)
    x = concatenate([left, right], axis=-1, name=f_name.format(fire_id, "concat"))
    return x


def SqueezeUNet2D(input_shape=None,
                  model_width=16,
                  deconv_ksize=2,
                  conv_activation='LeakyReLU'):
    inputs = Input(shape=(1, input_shape, 1))
    inputs_padding = ZeroPadding2D((1, 3))(inputs)
    """SqueezeUNet is a implementation based in SqueezeNetv1.1 and unet for semantic segmentation

    """
    if conv_activation == 'LeakyReLU':
        conv_activation = LeakyReLU()
    elif conv_activation == 'ReLU':
        conv_activation = ReLU()
    else:
        print('Do not support {}'.format(conv_activation))
        exit

    # CB1
    x01 = Conv2D(model_width, 3, strides=(3, 2), padding='same', activation=conv_activation, name='conv1')(
        inputs_padding)

    # CB2
    x02 = MaxPooling2D(pool_size=2, strides=2, name='pool1', padding='same')(x01)  # maxpooling 2 1

    # CB3
    x03 = fire_module(x02, fire_id=1, squeeze=model_width // 4, expand=model_width, conv_activation=conv_activation)
    x04 = fire_module(x03, fire_id=2, squeeze=model_width // 4, expand=model_width, conv_activation=conv_activation)
    x05 = MaxPooling2D(pool_size=2, strides=2, name='pool3', padding="same")(x04)

    # CB4
    x06 = fire_module(x05, fire_id=3, squeeze=(model_width * 2) // 4, expand=model_width * 2,
                      conv_activation=conv_activation)
    x07 = fire_module(x06, fire_id=4, squeeze=(model_width * 2) // 4, expand=model_width * 2,
                      conv_activation=conv_activation)
    x08 = MaxPooling2D(pool_size=2, strides=2, name='pool5', padding="same")(x07)

    # CB5
    x09 = fire_module(x08, fire_id=5, squeeze=(model_width * 3) // 4, expand=model_width * 3,
                      conv_activation=conv_activation)
    x10 = fire_module(x09, fire_id=6, squeeze=(model_width * 3) // 4, expand=model_width * 3,
                      conv_activation=conv_activation)
    x11 = fire_module(x10, fire_id=7, squeeze=(model_width * 4) // 4, expand=model_width * 4,
                      conv_activation=conv_activation)
    x12 = fire_module(x11, fire_id=8, squeeze=(model_width * 4) // 4, expand=model_width * 4,
                      conv_activation=conv_activation)

    up1 = concatenate(
        [Conv2DTranspose(model_width * 3, deconv_ksize, strides=1, padding='same', activation=conv_activation)(x12),
         x10], axis=-1)

    up2 = fire_module(up1, fire_id=9, squeeze=(model_width * 3) // 4, expand=model_width * 3,
                      conv_activation=conv_activation)

    up3 = concatenate(
        [Conv2DTranspose(model_width * 2, deconv_ksize, strides=1, padding='same', activation=conv_activation)(up2),
         x08], axis=-1)
    up4 = fire_module(up3, fire_id=10, squeeze=(model_width * 2) // 4, expand=model_width,
                      conv_activation=conv_activation)

    # EB1
    up5 = concatenate(
        [Conv2DTranspose(model_width, deconv_ksize, strides=(1, 2), padding='same', activation=conv_activation)(up4),
         x05], axis=-1)

    up6 = fire_module(up5, fire_id=11, squeeze=model_width // 4, expand=model_width, conv_activation=conv_activation)

    # EB2
    up7 = concatenate([Conv2DTranspose(model_width / 2, deconv_ksize, strides=(1, 2), padding='same',
                                       activation=conv_activation)(up6), x02], axis=-1)
    up8 = fire_module(up7, fire_id=12, squeeze=model_width // 4, expand=model_width // 2,
                      conv_activation=conv_activation)
    up9 = UpSampling2D(size=(1, 2))(up8)

    x = concatenate([up9, x01], axis=-1)
    x = Conv2D(model_width, 3, strides=1, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(1, 2))(x)
    x = Conv2D(1, (1, 3), activation=conv_activation)(x)
    with tf.compat.v1.variable_scope('collected') as scope:
        bbbbbx = x[:, :, :2500]
        bbbbxx = x[:, :, 3:2503]
        bbbxxx = x[:, :, 6:2506]
        bbxxxx = x[:, :, 10:2510]

    output = add((bbbbbx, bbbbxx, bbbxxx, bbxxxx), name='add')

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model


def SqueezeUNet2D_2(input_shape=None,
                    model_width=4,
                    deconv_ksize=2,
                    conv_activation='LeakyReLU'):
    inputs = Input(shape=input_shape) #(1, input_shape, 1))
    # inputs_padding = ZeroPadding2D((1, 8))(inputs)
    """SqueezeUNet is a implementation based in SqueezeNetv1.1 and unet for semantic segmentation

    """
    if conv_activation == 'LeakyReLU':
        conv_activation = LeakyReLU()
    elif conv_activation == 'ReLU':
        conv_activation = ReLU()
    else:
        print('Do not support {}'.format(conv_activation))
        exit

    # CB1
    x01 = Conv2D(model_width, 3, strides=(3, 5), padding='same', activation=conv_activation, name='conv1')(
        inputs)

    # CB2
    x02 = MaxPooling2D(pool_size=2, strides=(1, 2), name='pool1', padding='same')(x01)  # maxpooling 2 1

    # CB3
    x03 = fire_module(x02, fire_id=1, squeeze=model_width // 4, expand=model_width, conv_activation=conv_activation)
    x04 = fire_module(x03, fire_id=2, squeeze=model_width // 4, expand=model_width, conv_activation=conv_activation)
    x05 = MaxPooling2D(pool_size=2, strides=(1, 2), name='pool3', padding="same")(x04)

    # CB4
    x06 = fire_module(x05, fire_id=3, squeeze=(model_width * 2) // 4, expand=model_width * 2,
                      conv_activation=conv_activation)
    x07 = fire_module(x06, fire_id=4, squeeze=(model_width * 2) // 4, expand=model_width * 2,
                      conv_activation=conv_activation)
    x08 = MaxPooling2D(pool_size=2, strides=(1, 2), name='pool5', padding="same")(x07)

    # CB5
    x09 = fire_module(x08, fire_id=5, squeeze=(model_width * 3) // 4, expand=model_width * 3,
                      conv_activation=conv_activation)
    x10 = fire_module(x09, fire_id=6, squeeze=(model_width * 3) // 4, expand=model_width * 3,
                      conv_activation=conv_activation)
    x11 = fire_module(x10, fire_id=7, squeeze=(model_width * 4) // 4, expand=model_width * 4,
                      conv_activation=conv_activation)
    x12 = fire_module(x11, fire_id=8, squeeze=(model_width * 4) // 4, expand=model_width * 4,
                      conv_activation=conv_activation)

    up1 = concatenate(
        [Conv2DTranspose(model_width * 3, deconv_ksize, strides=1, padding='same', activation=conv_activation)(x12),
         x10], axis=-1)

    up2 = fire_module(up1, fire_id=9, squeeze=(model_width * 3) // 4, expand=model_width * 3,
                      conv_activation=conv_activation)

    up3 = concatenate(
        [Conv2DTranspose(model_width * 2, deconv_ksize, strides=1, padding='same', activation=conv_activation)(up2),
         x08], axis=-1)
    up4 = fire_module(up3, fire_id=10, squeeze=(model_width * 2) // 4, expand=model_width,
                      conv_activation=conv_activation)

    # EB1
    up5 = concatenate(
        [Conv2DTranspose(model_width, deconv_ksize, strides=(1, 2), padding='same', activation=conv_activation)(up4),
         x05], axis=-1)

    up6 = fire_module(up5, fire_id=11, squeeze=model_width // 4, expand=model_width, conv_activation=conv_activation)

    # EB2
    up7 = concatenate([Conv2DTranspose(model_width // 2, deconv_ksize, strides=(1, 2), padding='same', activation=conv_activation)(up6), x02], axis=-1)
    up8 = fire_module(up7, fire_id=12, squeeze=model_width // 4, expand=model_width // 2,
                      conv_activation=conv_activation)
    up9 = UpSampling2D(size=(1, 2))(up8)

    x = concatenate([up9, x01], axis=-1)
    x = Conv2D(model_width, 3, strides=1, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(1, 2))(x)

    output = Conv2D(1, (1, 3), padding='same', activation=conv_activation)(x)

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model

def SqueezeUNet2D_3(input_shape=None,
                    model_width=4,
                    deconv_ksize=2,
                    conv_activation='LeakyReLU'):
    inputs = Input(shape=input_shape) #(1, input_shape, 1))
    # inputs_padding = ZeroPadding2D((1, 8))(inputs)
    """SqueezeUNet is a implementation based in SqueezeNetv1.1 and unet for semantic segmentation

    """
    if conv_activation == 'LeakyReLU':
        conv_activation = LeakyReLU()
    elif conv_activation == 'ReLU':
        conv_activation = ReLU()
    else:
        print('Do not support {}'.format(conv_activation))
        exit

    # CB1
    x01 = Conv2D(model_width, 3, strides=(3, 5), padding='same', activation=conv_activation, name='conv1')(
        inputs)

    # CB2
    x02 = MaxPooling2D(pool_size=2, strides=(1, 4), name='pool1', padding='same')(x01)  # maxpooling 2 1

    # CB3
    x03 = fire_module(x02, fire_id=1, squeeze=model_width // 4, expand=model_width, conv_activation=conv_activation)
    x04 = fire_module(x03, fire_id=2, squeeze=model_width // 4, expand=model_width, conv_activation=conv_activation)
    x05 = MaxPooling2D(pool_size=2, strides=(1, 4), name='pool3', padding="same")(x04)

    # CB4
    x06 = fire_module(x05, fire_id=3, squeeze=(model_width * 2) // 4, expand=model_width * 2,
                      conv_activation=conv_activation)
    x07 = fire_module(x06, fire_id=4, squeeze=(model_width * 2) // 4, expand=model_width * 2,
                      conv_activation=conv_activation)
    x08 = MaxPooling2D(pool_size=2, strides=(1, 1), name='pool5', padding="same")(x07)

    up1 = concatenate(
            [Conv2DTranspose(model_width * 3, deconv_ksize, strides=1, padding='same', activation=conv_activation)(x08),
             x05], axis=-1)

    # # CB5
    # x09 = fire_module(x08, fire_id=5, squeeze=(model_width * 3) // 4, expand=model_width * 3,
    #                   conv_activation=conv_activation)
    # x10 = fire_module(x09, fire_id=6, squeeze=(model_width * 3) // 4, expand=model_width * 3,
    #                   conv_activation=conv_activation)
    # x11 = fire_module(x10, fire_id=7, squeeze=(model_width * 4) // 4, expand=model_width * 4,
    #                   conv_activation=conv_activation)
    # x12 = fire_module(x11, fire_id=8, squeeze=(model_width * 4) // 4, expand=model_width * 4,
    #                   conv_activation=conv_activation)
    #
    # up1 = concatenate(
    #     [Conv2DTranspose(model_width * 3, deconv_ksize, strides=1, padding='same', activation=conv_activation)(x12),
    #      x10], axis=-1)

    up2 = fire_module(up1, fire_id=9, squeeze=(model_width * 3) // 4, expand=model_width * 3,
                      conv_activation=conv_activation)

    up3 = concatenate(
        [Conv2DTranspose(model_width * 2, deconv_ksize, strides=1, padding='same', activation=conv_activation)(up2),
         x08], axis=-1)
    up4 = fire_module(up3, fire_id=10, squeeze=(model_width * 2) // 4, expand=model_width,
                      conv_activation=conv_activation)

    # EB1
    up5 = concatenate([up4, x05], axis=-2)

    up6 = fire_module(up5, fire_id=11, squeeze=model_width // 4, expand=model_width, conv_activation=conv_activation)

    # EB2
    up7 = UpSampling2D(size=(1, 2))(up6)
    up8 = concatenate([up7, x03], axis=-2)
    up9 = UpSampling2D(size=(1, 2))(up8)

    up10 = fire_module(up9, fire_id=12, squeeze=model_width // 4, expand=model_width // 2,
                      conv_activation=conv_activation)
    # up9 = UpSampling2D(size=(1, 4))(up8)

    y01 = concatenate([up10, x01], axis=-1)
    # x = Conv2D(model_width, 3, strides=1, padding='same', activation='relu')(x)
    y02 = UpSampling2D(size=(1, 5))(y01)

    output = Conv2D(1, (1, 3), padding='same', activation=conv_activation)(y02)

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model

def test_model():
    # FCNN(input_shape=2500)
    # AE_2D(input_shape=2500)
    SqueezeUNet2D_3(input_shape=(1, 1250, 1))


if __name__ == '__main__':
    test_model()
