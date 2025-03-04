import tensorflow as tf
import keras


def conv2d_net(x,
               num_filters,
               kernel_size,
               strides=1,
               pad='SAME',
               act=True,
               bn=True,
               rate=0.5,
               name=""):
    """

    """
    if bn:
        x = keras.layers.BatchNormalization(axis=-1, name=name + '_bn')(x)

    if act:
        x = keras.layers.ReLU(name=name + '_act')(x)

    if rate < 1.0:
        x = keras.layers.Dropout(rate=rate, name=name + '_drop')(x)

    x = keras.layers.Conv2D(filters=int(num_filters),
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=pad,
                            name=name + '_conv2d')(x)

    return x


def conv2d_net_squeeze_2(x,
                         num_filters,
                         kernel_size,
                         strides=1,
                         pad='SAME',
                         act=True,
                         bn=True,
                         rate=0.5,
                         name=""):
    """

    """
    if bn:
        x = keras.layers.BatchNormalization(axis=-1, name=name + '_bn')(x)

    if act:
        x = keras.layers.ReLU(name=name + '_act')(x)

    if rate < 1.0:
        x = keras.layers.Dropout(rate=rate, name=name + '_drop')(x)

    if x.shape[-1] <= num_filters:
        x = conv_squeeze(x, squeeze=8, expand=num_filters // 2, strides=1, name=name + '_conv2d_squeeze')

    else:
        x = keras.layers.Conv2D(filters=int(num_filters),
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=pad,
                                name=name + '_conv2d')(x)

    return x


def conv_squeeze(_x,
                 squeeze=16,
                 expand=64,
                 strides=2,
                 conv_activation=None,
                 name=""):
    f_name = "squeeze_{0}_{1}"
    # x = keras.layers.Conv1D(squeeze, 3, strides=1, activation=conv_activation, padding='same',
    #            name=f_name.format(name, "squeeze1"))(_x)
    x = keras.layers.BatchNormalization(axis=1)(_x)
    if "stage_3_conv2d" in name:
        a = 10

    left = keras.layers.Conv2D(expand, 1, strides=strides, activation=conv_activation, padding='same',
                               name=f_name.format(name, "expand1"))(x)

    if expand >= 8:
        right_left = keras.layers.Conv2D(expand // 2, 1, strides=strides, activation=conv_activation, padding='same',
                                         name=f_name.format(name, "expand3_l"))(x)
        right_right = keras.layers.Conv2D(expand // 2, 3, strides=strides, activation=conv_activation, padding='same',
                                          name=f_name.format(name, "expand3_r"))(x)
        right = keras.layers.concatenate([right_left, right_right], axis=-1, name=f_name.format(name, "concat_2"))
    else:
        right = keras.layers.Conv2D(expand, 3, strides=strides, activation=conv_activation, padding='same',
                                    name=f_name.format(name, "expand3"))(x)
    x = keras.layers.concatenate([left, right], axis=-1, name=f_name.format(name, "concat"))
    return x


def block1d_loop(xx, ff, stage, step):
    """

        :param xx:
        :param ff:
        :param stage:
        :param step:
        :return:
        """
    xx_skip = xx
    f1, f2 = ff
    # Batch norm, Activation, Dropout, Convolution (stride=1)
    xx = conv2d_net_squeeze_2(x=xx,
                              num_filters=f1,
                              kernel_size=3,
                              strides=1,
                              pad='SAME',
                              act=True,
                              bn=True,
                              rate=0.5,
                              name="resnet11a_{}_{}".format(step, stage))
    # Batch norm, Activation, Dropout, Convolution (stride=1)
    xx = conv2d_net_squeeze_2(x=xx,
                              num_filters=f2,
                              kernel_size=3,
                              strides=1,
                              pad='SAME',
                              act=True,
                              bn=True,
                              rate=0.5,
                              name="resnet11b_{}_{}".format(step, stage))

    xx = keras.layers.Add(name="skip11_{}_{}".format(step, stage))([xx, xx_skip])
    return xx


def selection_net(feature_len,
                  num_of_class=2,
                  from_logits=False,
                  filters_rhythm_net=None,
                  num_loop=9,
                  rate=0.5,
                  name='selection_net'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64),
                              (64, 80),
                              (80, 96),
                              (96, 112)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            tmp.append((max(f - filters_rhythm_net[0], filters_rhythm_net[0]), f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len * 3,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 3))(input_layer)
    # Convolution(stride=2)
    x = conv2d_net(x=resnet_input_layer,
                   num_filters=16,
                   kernel_size=3,
                   strides=2,
                   pad='SAME',
                   act=False,
                   bn=False,
                   rate=1.0,
                   name="input_stage")

    for st, ff in enumerate(filters_rhythm_net):
        st += 1
        f1, f2 = ff
        name = 'stage_{}'.format(st)
        # 1x1 Convolution (stride=2)
        x_skip = conv2d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv2d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=rate,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv2d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=rate,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    logits_layer = keras.layers.Dense(num_of_class)(x)
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq2_add_more2_250Hz(feature_len,
                                     num_of_class=2,
                                     from_logits=False,
                                     filters_rhythm_net=None,
                                     num_loop=3,
                                     rate=0.5,
                                     name='beat_concat_seq_add_more2_other', retrain=False):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [
            (16, 16),
            (16, 32),
            (32, 48),
            (48, 32)
        ]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(1, feature_len, 1))
    # resnet_input_layer = keras.layers.ZeroPadding2D(padding=(1, 6))(input_layer)
    # resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv2d_net(x=input_layer,
                   num_filters=filters_rhythm_net[0][0],
                   kernel_size=5,
                   strides=4,
                   pad='SAME',
                   act=False,
                   bn=False,
                   rate=1.0,
                   name="input_stage")

    for st, ff in enumerate(filters_rhythm_net):
        st += 1
        f1, f2 = ff
        name = 'stage_{}'.format(st)
        # 1x1 Convolution (stride=2)
        if st > 1:
            strides = 1
        else:
            strides = 3

        x_skip = conv2d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=strides,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv2d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=strides,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        print(f"{st} - {name}\n")

        x = conv2d_net_squeeze_2(x=x,
                                 num_filters=f2,
                                 kernel_size=3,
                                 strides=1,
                                 pad='SAME',
                                 act=True,
                                 bn=True,
                                 rate=0.5,
                                 name="resnet11_" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    if not retrain:
        x = conv2d_net(x=x,
                       num_filters=num_of_class,
                       kernel_size=4,
                       strides=1,
                       pad='SAME',
                       act=False,
                       bn=False,
                       rate=1.0,
                       name="pre_last_conv")

        bxxxxx = x[:, :, :80, :]
        bxxxx = x[:, :, 5:85, :]
        bxxx = x[:, :, 10:90, :]
        bxx = x[:, :, 15:95, :]
        bx = x[:, :, 20:100, :]
        b = x[:, :, 25:105, :]

        x = tf.concat((bxxxxx, bxxxx, bxxx, bxx, bx, b), axis=-1)

        logits_layer1 = keras.layers.Dense(num_of_class)(x)
        softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer1)
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        x = conv2d_net(x=x,
                       num_filters=num_of_class + 2,
                       kernel_size=4,
                       strides=2,
                       pad='SAME',
                       act=False,
                       bn=False,
                       rate=1.0,
                       name="last_conv")

        dense_layer = keras.layers.Dense(num_of_class, activation='relu', name='last_dense')(x)
        softmax_layer = keras.layers.Softmax(axis=-1, name='softmax')(dense_layer)
        train_model = keras.Model(input_layer, softmax_layer)

        return train_model


def beat_concat_seq3_250Hz(feature_len,
                           num_of_class=2,
                           from_logits=False,
                           filters_rhythm_net=None,
                           num_loop=3,
                           rate=0.5,
                           output_len=78,
                           name='beat_concat_seq3_250Hz', retrain=False):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [
            (16, 16),
            (16, 32),
            (32, 48),
            (48, 32)
        ]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(1, feature_len, 1))
    # resnet_input_layer = keras.layers.ZeroPadding2D(padding=(1, 6))(input_layer)
    # resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv2d_net(x=input_layer,
                   num_filters=filters_rhythm_net[0][0],
                   kernel_size=5,
                   strides=2,
                   pad='SAME',
                   act=False,
                   bn=False,
                   rate=1.0,
                   name="input_stage")

    for st, ff in enumerate(filters_rhythm_net):
        st += 1
        f1, f2 = ff
        name = 'stage_{}'.format(st)
        # 1x1 Convolution (stride=2)
        if st > 1:
            strides = 2
        else:
            strides = 2

        x_skip = conv2d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=strides,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv2d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=strides,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        # print(f"{st} - {name}\n")

        x = conv2d_net_squeeze_2(x=x,
                                 num_filters=f2,
                                 kernel_size=3,
                                 strides=1,
                                 pad='SAME',
                                 act=True,
                                 bn=True,
                                 rate=0.5,
                                 name="resnet11_" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    if not retrain:
        # x = conv2d_net(x=x,
        #                num_filters=num_of_class,
        #                kernel_size=4,
        #                strides=1,
        #                pad='SAME',
        #                act=False,
        #                bn=False,
        #                rate=1.0,
        #                name="pre_last_conv")

        x = conv2d_net(x=x,
                       num_filters=num_of_class,
                       kernel_size=(1,2),
                       strides=1,
                       pad='VALID',
                       act=False,
                       bn=False,
                       rate=1.0,
                       name="pre_last_conv")

        # x = tf.concat((x[:, :, :78, :], x[:, :, 1:, :]), axis=-1)

        logits_layer1 = keras.layers.Dense(num_of_class)(x)
        softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer1)
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        x = conv2d_net(x=x,
                       num_filters=num_of_class + 2,
                       kernel_size=4,
                       strides=2,
                       pad='SAME',
                       act=False,
                       bn=False,
                       rate=1.0,
                       name="last_conv")

        dense_layer = keras.layers.Dense(num_of_class, activation='relu', name='last_dense')(x)
        softmax_layer = keras.layers.Softmax(axis=-1, name='softmax')(dense_layer)
        train_model = keras.Model(input_layer, softmax_layer)

        return train_model

    # logits_layer1 = keras.layers.Dense(num_of_class)(x)
    # lstm_layer = keras.layers.Bidirectional(
    #     keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    # lstm_layer = keras.layers.Bidirectional(
    #     keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)
    #
    # logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)
    #
    # logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    # softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    #
    # if not from_logits:
    #     return keras.Model(input_layer, softmax_layer, name=name)
    # else:
    #     return keras.Model(input_layer, logits_layer, name=name)

def beat_concat_seq4_250Hz(feature_len,
                           num_of_class=2,
                           from_logits=False,
                           filters_rhythm_net=None,
                           num_loop=3,
                           rate=0.5,
                           output_len=78,
                           name='beat_concat_seq3_250Hz',
                           retrain=False):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [
            (16, 16),
            (16, 32),
            (32, 48),
            (48, 32)
        ]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(1, feature_len, 1))
    # resnet_input_layer = keras.layers.ZeroPadding2D(padding=(1, 6))(input_layer)
    # resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv2d_net(x=input_layer,
                   num_filters=filters_rhythm_net[0][0],
                   kernel_size=5,
                   strides=2,
                   pad='SAME',
                   act=False,
                   bn=False,
                   rate=1.0,
                   name="input_stage")

    for st, ff in enumerate(filters_rhythm_net):
        st += 1
        f1, f2 = ff
        name = 'stage_{}'.format(st)
        # 1x1 Convolution (stride=2)
        if st > 1:
            strides = 2
        else:
            strides = 2

        x_skip = conv2d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=strides,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv2d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=strides,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        # print(f"{st} - {name}\n")

        x = conv2d_net_squeeze_2(x=x,
                                 num_filters=f2,
                                 kernel_size=3,
                                 strides=1,
                                 pad='SAME',
                                 act=True,
                                 bn=True,
                                 rate=0.5,
                                 name="resnet11_" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    if not retrain:
        x = conv2d_net(x=x,
                       num_filters=num_of_class,
                       kernel_size=4,
                       strides=1,
                       pad='SAME',
                       act=False,
                       bn=False,
                       rate=1.0,
                       name="pre_last_conv")

        x = tf.concat((x[:, :, :output_len, :], x[:, :, 1:output_len + 1, :]), axis=-1)
        x = keras.layers.Flatten()(x)
        logits_layer1 = keras.layers.Dense(output_len*num_of_class)(x)
        softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer1)
        softmax_layer = tf.rehape(softmax_layer, [-1, output_len, num_of_class])
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        x = conv2d_net(x=x,
                       num_filters=num_of_class + 2,
                       kernel_size=4,
                       strides=2,
                       pad='SAME',
                       act=False,
                       bn=False,
                       rate=1.0,
                       name="last_conv")

        dense_layer = keras.layers.Dense(num_of_class, activation='relu', name='last_dense')(x)
        softmax_layer = keras.layers.Softmax(axis=-1, name='softmax')(dense_layer)
        train_model = keras.Model(input_layer, softmax_layer)

        return train_model


def beat_seq_mobilenet_v2_1d(feature_len,
                             num_of_class=2,
                             nu_from_logits=False,
                             nu_filters_rhythm_net=None,
                             nu_num_loop=0.5,
                             rate=0.5,
                             name='beat_seq_mobilenetv2_1d',
                             output_shape=100,
                             ):
    from models.mobilenet_v2 import MobileNetv2_1D

    return MobileNetv2_1D(feature_len, num_of_class, output_shape=output_shape, alpha=1.0, rate=rate, name=name)


def beat_seq_mobilenet_v2keras_1d(feature_len,
                                  num_of_class=2,
                                  nu_from_logits=False,
                                  nu_filters_rhythm_net=None,
                                  nu_num_loop=0.5,
                                  rate=0.5,
                                  name='beat_seq_mobilenetv2_1d',
                                  output_shape=100,
                                  ):
    from models.mobilenet_v2_keras import MobileNetV2_Keras_1D

    return MobileNetV2_Keras_1D(feature_len, num_of_class, output_shape=output_shape, alpha=1.0, rate=rate, name=name)


def beat_concat_seq_add_depthwise_250Hz(feature_len,
                                        num_of_class=2,
                                        from_logits=False,
                                        filters_rhythm_net=None,
                                        num_loop=7,
                                        rate=0.5,
                                        name='beat_concat_seq_add_more2_other'):
    from models.model_depthwise import beat_concat_seq_add_depthwise_250Hz

    return beat_concat_seq_add_depthwise_250Hz(feature_len,
                                               num_of_class=num_of_class,
                                               from_logits=from_logits,
                                               filters_rhythm_net=filters_rhythm_net,
                                               num_loop=num_loop,
                                               rate=rate,
                                               name='beat_concat_seq_add_more2_other')


def beat_depthwise_250Hz(feature_len,
                         num_of_class=2,
                         from_logits=False,
                         filters_rhythm_net=None,
                         num_loop=7,
                         rate=0.5,
                         name='beat_concat_seq_add_more2_other'):
    from models.model_depthwise import beat_depthwise_250Hz

    return beat_depthwise_250Hz(feature_len,
                                num_of_class=num_of_class,
                                from_logits=from_logits,
                                filters_rhythm_net=filters_rhythm_net,
                                num_loop=num_loop,
                                rate=rate,
                                name='beat_concat_seq_add_more2_other')


def beat_depthwise2_128Hz(feature_len,
                          num_of_class=2,
                          from_logits=False,
                          filters_rhythm_net=None,
                          num_loop=7,
                          rate=0.5,
                          name='beat_concat_seq_add_more2_other'):
    from models.model_depthwise import beat_depthwise2_128Hz

    return beat_depthwise2_128Hz(feature_len,
                                 num_of_class=num_of_class,
                                 name=name)


def beat_squeezeunet_128Hz(feature_len,
                           num_of_class=3,
                           from_logits=False,
                           filters_rhythm_net=None,
                           num_loop=7,
                           rate=0.5,
                           name='beat_squeezeunet_128Hz'):
    from models.unet_1D import SqueezeUNet
    return SqueezeUNet(feature_len=feature_len,
                       num_of_class=num_of_class,
                       name=name)


def test_model():
    feature_len = 640
    num_of_class = 2
    from_logits = False,
    filters_rhythm_net = None,
    num_loop = 7
    rate = 0.5
    # model = beat_concat_seq_add_more2_128Hz(feature_len=feature_len,
    #                               num_of_class=num_of_class)
    # model = beat_concat_seq2_add_more2_128Hz(feature_len=640,
    model = beat_concat_seq3_250Hz(feature_len=1250,
                                   # model = beat_depthwise2_128Hz(feature_len=640,
                                   # model = beat_concat_sequeeze_add_more2_128Hz(feature_len=640,
                                   num_of_class=4,
                                   from_logits=False,
                                   filters_rhythm_net=[8, 24, 8],  #[8, 16, 32],
                                   num_loop=2,
                                   rate=0.5,
                                   name='beat_concat_seq_add_more2_other')

    model.summary()

    # import numpy as np
    # label = np.random.randint(2, size=(100, 1, 78, 4))
    # sample = np.random.randint(10, size=(100, 1, 1250, 1))
    #
    # model.compile(optimizer='adam', loss='binary_crossentropy')
    # model.fit(x=sample, y=label, epochs=1)


def freeze_model(train_model, num_class=2, last_layer_name='last_conv_conv1d'):
    ######### FREEZE MODEL #############
    input_layer = train_model.input
    last_layer = train_model.get_layer(last_layer_name).output

    dense_layer = keras.layers.Dense(num_class, activation='relu', name='last_dense')(last_layer)
    softmax_layer = keras.layers.Softmax(axis=-1, name='softmax')(dense_layer)
    train_model = keras.Model(input_layer, softmax_layer)

    for i_layer in range(len(train_model.layers) - 2):
        train_model.layers[i_layer].trainable = False

    return train_model


if __name__ == '__main__':
    test_model()
