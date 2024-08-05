import warnings

from keras import backend
from keras import layers
from keras.applications import imagenet_utils
from keras.models import Model

BASE_WEIGHT_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/"
)


def MobileNetV2_Keras_1D(input_shape=None,
                         classes=5,
                         output_shape=None,
                         alpha=1.0,
                         rate=0.5,
                         name='MobileNetv2_1D',
                         classifier_activation="softmax",
                         channel_axis = 1
                         ):
    inputs = layers.Input(shape=(input_shape, 1))

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.Conv1D(
        first_block_filters,
        kernel_size=3,
        strides=3,
        padding="same",
        use_bias=False,
        name="Conv1",
    )(inputs)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="bn_Conv1"
    )(x)
    x = layers.ReLU(6.0, name="Conv1_relu")(x)

    x = _inverted_res_block(
        x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0
    )

    x = _inverted_res_block(
        x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1
    )
    x = _inverted_res_block(
        x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2
    )

    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3
    )
    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4
    )
    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5
    )

    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6
    )
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7
    )
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8
    )
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9
    )

    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10
    )
    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11
    )
    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12
    )

    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=13
    )
    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14
    )
    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15
    )

    x = _inverted_res_block(
        x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16
    )

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we increase the number of output
    # channels.
    if alpha > 1.0:
        last_block_filters = _make_divisible(output_shape * alpha, 8)
    else:
        last_block_filters = output_shape

    x = layers.Conv1D(
        last_block_filters, kernel_size=1, use_bias=False, name="Conv_1")(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv_1_bn")(x)
    x = layers.ReLU(6.0, name="out_relu")(x)

    # x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(
        classes, activation=classifier_activation, name="predictions"
    )(x)

    bbbxx = x[:, :-3, :]
    bbxx = x[:, 1:-2, :]
    bxx = x[:, 2:-1, :]
    xx = x[:, 3:, :]

    x = layers.Concatenate(axis=2)([bbbxx, bbxx, bxx, xx])

    logits_layer1 = layers.Dense(classes)(x)
    lstm_layer = layers.Bidirectional(
        layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(x)
    lstm_layer = layers.Bidirectional(
        layers.LSTM(x.shape[-1], return_sequences=True, dropout=rate))(lstm_layer)

    logits_layer2 = layers.Dense(classes)(lstm_layer)
    logits_layer = layers.Add()([logits_layer1, logits_layer2])

    output = layers.Activation('softmax', name='softmax')(logits_layer)

    model = Model(inputs, output, name=f"mobilenetv2_{alpha:0.2f}_{input_shape}")

    return model


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    """Inverted ResNet block."""
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    in_channels = inputs.shape[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    # Ensure the number of filters on the last 1x1 convolution is divisible by
    # 8.
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = f"block_{block_id}_"

    if block_id:
        # Expand with a pointwise 1x1 convolution.
        x = layers.Conv1D(
            expansion * in_channels,
            kernel_size=1,
            padding="same",
            use_bias=False,
            activation=None,
            name=prefix + "expand",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "expand_BN",
        )(x)
        x = layers.ReLU(6.0, name=prefix + "expand_relu")(x)
    else:
        prefix = "expanded_conv_"

    # Depthwise 3x3 convolution.
    # if stride == 2:
    #     x = layers.ZeroPadding1D(
    #         padding='same', name=prefix + "pad"
    #     )(x)
    x = layers.DepthwiseConv1D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding="same" if stride == 1 else "valid",
        name=prefix + "depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "depthwise_BN",
    )(x)

    x = layers.ReLU(6.0, name=prefix + "depthwise_relu")(x)

    # Project with a pointwise 1x1 convolution.
    x = layers.Conv1D(
        pointwise_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        activation=None,
        name=prefix + "project",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project_BN",
    )(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + "add")([inputs, x])
    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

if __name__ == '__main__':
    model = MobileNetV2_Keras_1D(2500, 5, 100, 1.0)
    print(model.summary())