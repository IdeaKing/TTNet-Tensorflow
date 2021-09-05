# Created by Thomas Chia
# Based on the TTNet Architecture
# Changes made on the orginal paper:
# - No local ball detection stage
# DOC: 2021-08-02

from tensorflow.keras.layers import (Conv2D, ReLU, Flatten, Dense,
                                     BatchNormalization, MaxPool2D,
                                     Dropout, Conv2DTranspose)
from tensorflow.keras import (Input, Model)
from utils.configs import configs


def conv_block(layer, filters, maxpool=True):
    """The ConvBlock in TTNet"""
    x = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=1,
        padding="same")(layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    if maxpool:
        x = MaxPool2D(strides=2, padding="valid")(x)
    return x


def ball_detection(input, dropout, output):
    """Ball detection stage."""
    conv1 = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        strides=1,
        padding="same")(input)
    bnorm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bnorm1)
    block1 = conv_block(layer=relu1, filters=64)
    block2 = conv_block(layer=block1, filters=64)
    drop1 = Dropout(rate=dropout)(block2)
    block3 = conv_block(layer=drop1, filters=128)
    block4 = conv_block(layer=block3, filters=128)
    drop2 = Dropout(rate=dropout)(block4)
    block5 = conv_block(layer=drop2, filters=256)
    block6 = conv_block(layer=block5, filters=256)
    drop3 = Dropout(rate=dropout)(block6)
    flat = Flatten()(drop3)
    fc1 = Dense(
        units=(2* (output[0]+output[1])), 
        activation="linear")(flat)
    relu2 = ReLU()(fc1)
    drop4 = Dropout(rate=dropout)(relu2)
    # Swallow Tail Shape X
    fc2_x = Dense(
        units=(output[0]*2), 
        activation="linear")(drop4)
    relu3_x = ReLU()(fc2_x)
    drop5_x = Dropout(rate=dropout)(relu3_x)
    fc3_x = Dense(
        units=output[0], 
        activation="sigmoid")(drop5_x)
    # Swallow Tail Shape Y
    fc2_y = Dense(
        units=(output[1]*2), 
        activation="linear")(drop4)
    relu3_y = ReLU()(fc2_y)
    drop5_y = Dropout(rate=dropout)(relu3_y)
    fc3_y = Dense(
        units=output[1], 
        activation="sigmoid")(drop5_y)
    return fc3_x, fc3_y, block2, block3, block4, block5, block6


def deconv_block(layer, filters):
    """The DeconvBlock in TTNet"""
    x = Conv2D(
        filters=filters/2,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid")(layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(
        filters=filters/2,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def sem_segmentation(inputs):
    """Semantic segmentation stage."""
    dblock5 = deconv_block(layer=inputs[3], filters=128)
    dblock5 = dblock5 + inputs[2]
    dblock4 = deconv_block(layer=dblock5, filters=128)
    dblock4 = dblock4 + inputs[1]
    dblock3 = deconv_block(layer=dblock4, filters=64)
    dblock3 = dblock3 + inputs[0]
    dblock2 = deconv_block(layer=dblock3, filters=64)
    tconv1 = Conv2DTranspose(
        filters=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid")(dblock2)
    relu1 = ReLU()(tconv1)
    conv1 = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=1,
        padding="same")(relu1)
    relu2 = ReLU()(conv1)
    conv2 = Conv2D(
        filters=3,
        kernel_size=(2, 2),
        strides=1,
        padding="valid",
        activation="sigmoid")(relu2)
    return conv2


def event_spotting(input, dropout):
    """Event spotting stage."""
    conv1 = Conv2D(
        filters=64,
        kernel_size=(1, 1))(input)
    bnorm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bnorm1)
    drop1 = Dropout(rate=dropout)(relu1)
    block1 = conv_block(layer=drop1, filters=64, maxpool=False)
    drop2 = Dropout(rate=dropout)(block1)
    block2 = conv_block(layer=drop2, filters=64, maxpool=False)
    drop3 = Dropout(rate=dropout)(block2)
    flat = Flatten()(drop3)
    fc1 = Dense(units=512, activation="linear")(flat)
    relu2 = ReLU()(fc1)
    fc2 = Dense(units=2, activation="sigmoid")(relu2)
    return fc2


def ttnet(dims, dropout=0.1, ball_detection_stage=False):
    """TTNet Model.
    Notes:
        Ball detection stage is a requirement.
    """

    if ball_detection_stage==True:
        input = Input(shape=(dims[0], dims[1], dims[2]))
        detection_x, detection_y, block2, block3, block4, block5, block6 = ball_detection(
            input=input, 
            dropout=dropout, 
            output=configs.processed_image_shape)
        model = Model(inputs=[input], outputs=[detection_x, detection_y])
    else:
        input = Input(shape=(dims[0], dims[1], dims[2]))
        detection_x, detection_y, block2, block3, block4, block5, block6 = ball_detection(
            input=input, dropout=dropout)
        mask = sem_segmentation(inputs=[block2, block3, block4, block5])
        events = event_spotting(input=block6, dropout=dropout)
        model = Model(inputs=[input], outputs=[detection_x, detection_y, events, mask])

    return model


if __name__ == "__main__":
    """Test and compile the model."""
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except IOError:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

    model = ttnet(dims=(320, 128, 27), dropout=.2, ball_detection_stage=True)
    model.summary()
