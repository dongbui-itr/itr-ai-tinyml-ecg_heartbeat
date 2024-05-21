import tensorflow as tf
import keras


def conv1d_net(x,
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

    x = keras.layers.Conv1D(filters=int(num_filters),
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=pad,
                            name=name + '_conv1d')(x)

    return x


def conv1d_net_squeeze(x,
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
    if "resnet11a_0_stage_3" in name:
        a = 10

    if bn:
        x = keras.layers.BatchNormalization(axis=-1, name=name + '_bn')(x)

    if act:
        x = keras.layers.ReLU(name=name + '_act')(x)

    if rate < 1.0:
        x = keras.layers.Dropout(rate=rate, name=name + '_drop')(x)

    x = keras.layers.Conv1D(filters=int(num_filters),
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=pad,
                            name=name + '_conv1d')(x)

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

    left = keras.layers.Conv1D(expand, 1, strides=strides, activation=conv_activation, padding='same',
                               name=f_name.format(name, "expand1"))(x)
    right = keras.layers.Conv1D(expand, 3, strides=strides, activation=conv_activation, padding='same',
                                name=f_name.format(name, "expand3"))(x)
    x = keras.layers.concatenate([left, right], axis=2, name=f_name.format(name, "concat"))
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
    xx = conv1d_net(x=xx,
                    num_filters=f1,
                    kernel_size=3,
                    strides=1,
                    pad='SAME',
                    act=True,
                    bn=True,
                    rate=0.5,
                    name="resnet11a_{}_{}".format(step, stage))
    # Batch norm, Activation, Dropout, Convolution (stride=1)
    xx = conv1d_net(x=xx,
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
    x = conv1d_net(x=resnet_input_layer,
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=rate,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
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


def rhythm_net(feature_len,
               num_of_class=2,
               from_logits=False,
               filters_rhythm_net=None,
               num_loop=9,
               rate=0.5,
               name='rhythm_net'):
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

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=rate,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
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


def rhythm_seq(feature_len,
               num_of_class=2,
               from_logits=False,
               filters_rhythm_net=None,
               num_loop=9,
               rate=0.5,
               name='rhythm_seq'):
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

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=rate,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
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

    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer = keras.layers.Dense(num_of_class)(lstm_layer)
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def rhythm_seq_add(feature_len,
                   num_of_class=2,
                   from_logits=False,
                   filters_rhythm_net=None,
                   num_loop=9,
                   rate=0.5,
                   name='rhythm_seq_add'):
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

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=rate,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
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

    logits_layer1 = keras.layers.Dense(num_of_class)(x)

    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)
    logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)

    logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_net(feature_len,
             num_of_class=2,
             from_logits=False,
             filters_rhythm_net=None,
             num_loop=7,
             rate=0.5,
             name='beat_net'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=rate,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
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


def beat_seq(feature_len,
             num_of_class=2,
             from_logits=False,
             filters_rhythm_net=None,
             num_loop=7,
             rate=0.5,
             name='beat_seq'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer = keras.layers.Dense(num_of_class)(lstm_layer)
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_seq_add(feature_len,
                 num_of_class=2,
                 from_logits=False,
                 filters_rhythm_net=None,
                 num_loop=7,
                 rate=0.5,
                 name='beat_seq_add'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    logits_layer1 = keras.layers.Dense(num_of_class)(x)

    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)

    logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq(feature_len,
                    num_of_class=2,
                    from_logits=False,
                    filters_rhythm_net=None,
                    num_loop=7,
                    rate=0.5,
                    name='beat_concat_seq'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    with tf.compat.v1.variable_scope('collected') as scope:
        xx = x[:, 1:, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        xx = tf.concat((xx, zz), axis=1)

        yy = x[:, 2:, :]
        zz = tf.zeros_like(x[:, 0:2, :])
        yy = tf.concat((yy, zz), axis=1)

        xy = x[:, 3:, :]
        zz = tf.zeros_like(x[:, 0:3, :])
        xy = tf.concat((xy, zz), axis=1)

        x = tf.concat((x, xx, yy, xy), axis=2)

    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer = keras.layers.Dense(num_of_class)(lstm_layer)
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq_other(feature_len,
                          num_of_class=2,
                          from_logits=False,
                          filters_rhythm_net=None,
                          num_loop=7,
                          rate=0.5,
                          name='beat_concat_seq'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    with tf.compat.v1.variable_scope('collected') as scope:
        xx = x[:, :-1, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        xx = tf.concat((zz, xx), axis=1)

        yy = x[:, 1:, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        yy = tf.concat((yy, zz), axis=1)

        x = tf.concat((xx, x, yy), axis=2)

    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer = keras.layers.Dense(num_of_class)(lstm_layer)
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq_other2(feature_len,
                           num_of_class=2,
                           from_logits=False,
                           filters_rhythm_net=None,
                           num_loop=7,
                           rate=0.5,
                           name='beat_concat_seq'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    with tf.compat.v1.variable_scope('collected') as scope:
        xx = x[:, :-1, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        xx = tf.concat((zz, xx), axis=1)

        yy = x[:, 1:, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        yy = tf.concat((yy, zz), axis=1)

    lstm_layer_x = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer_x = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer_x)

    lstm_layer_xx = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(xx)
    lstm_layer_xx = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer_xx)

    lstm_layer_yy = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(yy)
    lstm_layer_yy = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer_yy)

    lstm_layer = tf.concat((lstm_layer_xx, lstm_layer_x, lstm_layer_yy), axis=2)

    logits_layer = keras.layers.Dense(num_of_class)(lstm_layer)
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq_add(feature_len,
                        num_of_class=2,
                        from_logits=False,
                        filters_rhythm_net=None,
                        num_loop=7,
                        rate=0.5,
                        name='beat_concat_seq_add'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    with tf.compat.v1.variable_scope('collected') as scope:
        xx = x[:, 1:, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        xx = tf.concat((xx, zz), axis=1)

        yy = x[:, 2:, :]
        zz = tf.zeros_like(x[:, 0:2, :])
        yy = tf.concat((yy, zz), axis=1)

        xy = x[:, 3:, :]
        zz = tf.zeros_like(x[:, 0:3, :])
        xy = tf.concat((xy, zz), axis=1)

        x = tf.concat((x, xx, yy, xy), axis=2)

    logits_layer1 = keras.layers.Dense(num_of_class)(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)

    logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq_add_other(feature_len,
                              num_of_class=2,
                              from_logits=False,
                              filters_rhythm_net=None,
                              num_loop=7,
                              rate=0.5,
                              name='beat_concat_seq_add'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    with tf.compat.v1.variable_scope('collected') as scope:
        xx = x[:, :-1, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        xx = tf.concat((zz, xx), axis=1)

        yy = x[:, 1:, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        yy = tf.concat((yy, zz), axis=1)

        x = tf.concat((xx, x, yy), axis=2)

    logits_layer1 = keras.layers.Dense(num_of_class)(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)

    logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq_add_other1(feature_len,
                               num_of_class=2,
                               from_logits=False,
                               filters_rhythm_net=None,
                               num_loop=7,
                               rate=0.5,
                               name='beat_concat_seq_add'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    with tf.compat.v1.variable_scope('collected') as scope:
        aa = x[:, :-2, :]
        zz = tf.zeros_like(x[:, 0:2, :])
        aa = tf.concat((zz, aa), axis=1)

        xx = x[:, :-1, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        xx = tf.concat((zz, xx), axis=1)

        yy = x[:, 1:, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        yy = tf.concat((yy, zz), axis=1)

        bb = x[:, 2:, :]
        zz = tf.zeros_like(x[:, 0:2, :])
        bb = tf.concat((bb, zz), axis=1)

        x = tf.concat((aa, xx, x, yy, bb), axis=2)

    logits_layer1 = keras.layers.Dense(num_of_class)(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)

    logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq_add_other2(feature_len,
                               num_of_class=2,
                               from_logits=False,
                               filters_rhythm_net=None,
                               num_loop=7,
                               rate=0.5,
                               name='beat_concat_seq'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    with tf.compat.v1.variable_scope('collected') as scope:
        xx = x[:, :-1, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        xx = tf.concat((zz, xx), axis=1)

        yy = x[:, 1:, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        yy = tf.concat((yy, zz), axis=1)

    lstm_layer_x = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer_x = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer_x)

    lstm_layer_xx = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(xx)
    lstm_layer_xx = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer_xx)

    lstm_layer_yy = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(yy)
    lstm_layer_yy = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer_yy)

    lstm_layer = keras.layers.Add()([lstm_layer_xx, lstm_layer_x, lstm_layer_yy])

    logits_layer = keras.layers.Dense(num_of_class)(lstm_layer)
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seqn_add(feature_len,
                         num_of_class=2,
                         from_logits=False,
                         filters_rhythm_net=None,
                         num_loop=7,
                         rate=0.5,
                         name='beat_concat_seqn_add'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    with tf.compat.v1.variable_scope('collected') as scope:
        xx = x[:, 1:, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        xx = tf.concat((xx, zz), axis=1)

        yy = x[:, 2:, :]
        zz = tf.zeros_like(x[:, 0:2, :])
        yy = tf.concat((yy, zz), axis=1)

        xy = x[:, 3:, :]
        zz = tf.zeros_like(x[:, 0:3, :])
        xy = tf.concat((xy, zz), axis=1)

        x = tf.concat((x, xx, yy, xy), axis=2)

    logits_layer1 = keras.layers.Dense(num_of_class)(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True))(lstm_layer)

    logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)

    logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq2_add(feature_len,
                         num_of_class=2,
                         from_logits=False,
                         filters_rhythm_net=None,
                         num_loop=7,
                         rate=0.5,
                         name='beat_concat_seq2_add'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    with tf.compat.v1.variable_scope('collected') as scope:
        xx = x[:, 1:, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        xx = tf.concat((xx, zz), axis=1)

        yy = x[:, 2:, :]
        zz = tf.zeros_like(x[:, 0:2, :])
        yy = tf.concat((yy, zz), axis=1)

        xy = x[:, 3:, :]
        zz = tf.zeros_like(x[:, 0:3, :])
        xy = tf.concat((xy, zz), axis=1)

        x = tf.concat((x, xx, yy, xy), axis=2)

    logits_layer1 = keras.layers.Dense(num_of_class)(x)

    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)

    logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq3_add(feature_len,
                         num_of_class=2,
                         from_logits=False,
                         filters_rhythm_net=None,
                         num_loop=7,
                         rate=0.5,
                         name='beat_concat_seq3_add'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    with tf.compat.v1.variable_scope('collected') as scope:
        xx = x[:, 1:, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        xx = tf.concat((xx, zz), axis=1)

        yy = x[:, 2:, :]
        zz = tf.zeros_like(x[:, 0:2, :])
        yy = tf.concat((yy, zz), axis=1)

        xy = x[:, 3:, :]
        zz = tf.zeros_like(x[:, 0:3, :])
        xy = tf.concat((xy, zz), axis=1)

        x = tf.concat((x, xx, yy, xy), axis=2)

    logits_layer1 = keras.layers.Dense(num_of_class)(x)

    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)

    logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq_add_more_other(feature_len,
                                   num_of_class=2,
                                   from_logits=False,
                                   filters_rhythm_net=None,
                                   num_loop=7,
                                   rate=0.5,
                                   name='beat_concat_seq_add_more_other'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    with tf.compat.v1.variable_scope('collected') as scope:
        bxx = x[:, :-2, :]
        bzz = tf.zeros_like(x[:, 0:2, :])
        bxx = tf.concat((bzz, bxx), axis=1)

        xx = x[:, :-1, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        xx = tf.concat((zz, xx), axis=1)

        yy = x[:, 1:, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        yy = tf.concat((yy, zz), axis=1)

        ayy = x[:, 2:, :]
        azz = tf.zeros_like(x[:, 0:2, :])
        ayy = tf.concat((ayy, azz), axis=1)

        x = tf.concat((bxx, xx, x, yy, ayy), axis=2)

    logits_layer1 = keras.layers.Dense(num_of_class)(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)

    logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq_add_more2_other(feature_len,
                                    num_of_class=2,
                                    from_logits=False,
                                    filters_rhythm_net=None,
                                    num_loop=7,
                                    rate=0.5,
                                    name='beat_concat_seq_add_more2_other'):
    """

    """
    if filters_rhythm_net is None:
        filters_rhythm_net = [(16, 16),
                              (16, 32),
                              (32, 48),
                              (48, 64)]
    else:
        tmp = []
        for i, f in enumerate(filters_rhythm_net):
            if len(tmp) == 0:
                tmp.append((f, f))
            else:
                tmp.append((tmp[-1][-1], f))

        filters_rhythm_net = tmp.copy()

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    with tf.compat.v1.variable_scope('collected') as scope:
        bbxx = x[:, :-3, :]
        bbzz = tf.zeros_like(x[:, 0:3, :])
        bbxx = tf.concat((bbzz, bbxx), axis=1)

        bxx = x[:, :-2, :]
        bzz = tf.zeros_like(x[:, 0:2, :])
        bxx = tf.concat((bzz, bxx), axis=1)

        xx = x[:, :-1, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        xx = tf.concat((zz, xx), axis=1)

        yy = x[:, 1:, :]
        zz = tf.zeros_like(x[:, 0:1, :])
        yy = tf.concat((yy, zz), axis=1)

        ayy = x[:, 2:, :]
        azz = tf.zeros_like(x[:, 0:2, :])
        ayy = tf.concat((ayy, azz), axis=1)

        aayy = x[:, 3:, :]
        aazz = tf.zeros_like(x[:, 0:3, :])
        aayy = tf.concat((aayy, aazz), axis=1)

        x = tf.concat((bbxx, bxx, xx, x, yy, ayy, aayy), axis=2)

    logits_layer1 = keras.layers.Dense(num_of_class)(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)

    logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)
    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_seq_add_more2_128Hz(feature_len,
                                    num_of_class=2,
                                    from_logits=False,
                                    filters_rhythm_net=None,
                                    num_loop=3,
                                    rate=0.5,
                                    name='beat_concat_seq_add_more2_other'):
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

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
                   kernel_size=4,
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net_squeeze(x=x,
                               num_filters=f2,
                               kernel_size=3,
                               strides=1,
                               pad='SAME',
                               act=True,
                               bn=True,
                               rate=0.5,
                               name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    # with tf.compat.v1.variable_scope('collected') as scope:
    #     bbxx = x[:, :-5, :]
    #     bxx = x[:, 2:-3, :]
    #     xx = x[:, 5:, :]
    #
    #     x = tf.concat((bbxx, bxx, xx), axis=2)
    #
    #     # yy = x[:, 1:, :]
    #     # zz = tf.zeros_like(x[:, 0:1, :])
    #     # yy = tf.concat((yy, zz), axis=1)
    #     #
    #     # ayy = x[:, 2:, :]
    #     # azz = tf.zeros_like(x[:, 0:2, :])
    #     # ayy = tf.concat((ayy, azz), axis=1)
    #     #
    #     # aayy = x[:, 3:, :]
    #     # aazz = tf.zeros_like(x[:, 0:3, :])
    #     # aayy = tf.concat((aayy, aazz), axis=1)
    #     #
    #     # x = tf.concat((bbxx, bxx, xx, x, yy, ayy, aayy), axis=2)

    x = conv1d_net(x=x,
                   num_filters=x.shape[-1],
                   kernel_size=4,
                   strides=2,
                   pad='SAME',
                   act=False,
                   bn=False,
                   rate=1.0,
                   name="last_conv")

    logits_layer1 = keras.layers.Dense(num_of_class)(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)

    logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)

    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


def beat_concat_sequeeze_add_more2_128Hz(feature_len,
                                         num_of_class=2,
                                         from_logits=False,
                                         filters_rhythm_net=None,
                                         num_loop=3,
                                         rate=0.5,
                                         name='beat_concat_seq_add_more2_other'):
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

    input_layer = keras.layers.Input(shape=(feature_len,))
    resnet_input_layer = keras.layers.Reshape((feature_len, 1))(input_layer)
    # Convolution(stride=2)
    x = conv1d_net(x=resnet_input_layer,
                   num_filters=filters_rhythm_net[0][0],
                   kernel_size=4,
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
        x_skip = conv1d_net(x=x,
                            num_filters=f2,
                            kernel_size=1,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="skip12_" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=2)
        x = conv1d_net(x=x,
                       num_filters=f1,
                       kernel_size=3,
                       strides=2,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet12" + name)
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        x = conv1d_net(x=x,
                       num_filters=f2,
                       kernel_size=3,
                       strides=1,
                       pad='SAME',
                       act=True,
                       bn=True,
                       rate=0.5,
                       name="resnet11" + name)

        x = keras.layers.Add(name="add_" + name)([x, x_skip])
        ffs = [(f2, f2) for _ in range(num_loop)]
        for sl, ffl in enumerate(ffs):
            x = block1d_loop(x, ffl, name, sl)

    # with tf.compat.v1.variable_scope('collected') as scope:
    #     bbxx = x[:, :-5, :]
    #     bxx = x[:, 2:-3, :]
    #     xx = x[:, 5:, :]
    #
    #     x = tf.concat((bbxx, bxx, xx), axis=2)
    #
    #     # yy = x[:, 1:, :]
    #     # zz = tf.zeros_like(x[:, 0:1, :])
    #     # yy = tf.concat((yy, zz), axis=1)
    #     #
    #     # ayy = x[:, 2:, :]
    #     # azz = tf.zeros_like(x[:, 0:2, :])
    #     # ayy = tf.concat((ayy, azz), axis=1)
    #     #
    #     # aayy = x[:, 3:, :]
    #     # aazz = tf.zeros_like(x[:, 0:3, :])
    #     # aayy = tf.concat((aayy, aazz), axis=1)
    #     #
    #     # x = tf.concat((bbxx, bxx, xx, x, yy, ayy, aayy), axis=2)

    # x1 = conv1d_net(x=x,
    #                num_filters=x.shape[-1],
    #                kernel_size=4,
    #                strides=2,
    #                pad='SAME',
    #                act=False,
    #                bn=False,
    #                rate=1.0,
    #                name="last_conv")
    #
    # print(x1.shape)

    x = conv_squeeze(x,
                     squeeze=8,
                     expand=8,
                     strides=2,
                     conv_activation=None,
                     name="last_conv")

    logits_layer1 = keras.layers.Dense(num_of_class)(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = keras.layers.Bidirectional(
        keras.layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer2 = keras.layers.Dense(num_of_class)(lstm_layer)

    logits_layer = keras.layers.Add()([logits_layer1, logits_layer2])
    softmax_layer = keras.layers.Softmax(axis=-1)(logits_layer)

    if not from_logits:
        return keras.Model(input_layer, softmax_layer, name=name)
    else:
        return keras.Model(input_layer, logits_layer, name=name)


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
    model = beat_concat_seq_add_more2_128Hz(feature_len=640,
                                            # model = beat_concat_sequeeze_add_more2_128Hz(feature_len=640,
                                            num_of_class=3,
                                            from_logits=False,
                                            filters_rhythm_net=[4, 8, 16],
                                            num_loop=3,
                                            rate=0.5,
                                            name='beat_concat_seq_add_more2_other')

    model.summary()


if __name__ == '__main__':
    test_model()
