from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf


class UNet_segmentation:
    """ UNet architecture from following github notebook for image segmentation:
        https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb

        1 963 043 parameters (4,589,283 parameters with embeding from MobileNet in bottleneck layer)
    """

    def __init__(self, use_embeding=True, input_size=(256, 256, 3), filters=(16, 32, 64, 128, 256)):

        # TODO - different losses

        p0 = Input(input_size)

        if use_embeding:
            mobilenet_model = tf.keras.applications.MobileNetV2(input_shape=(256, 256, 3),
                                                                include_top=False,
                                                                weights='imagenet')
            mobilenet_model.trainable = False
            for layer in mobilenet_model.layers:
                layer.trainable = False

            mn1 = mobilenet_model(p0)
            mn1 = Reshape((16, 16, 320))(mn1)

        conv_outputs = []
        int_layer = p0

        for f in filters:
            conv_output, int_layer = UNet_segmentation.down_block(int_layer, f)
            conv_outputs.append(conv_output)

        int_layer = UNet_segmentation.bottleneck(int_layer, filters[-1])

        if use_embeding:
            int_layer = Concatenate()([int_layer, mn1])

        conv_outputs = list(reversed(conv_outputs))
        for i, f in enumerate(reversed(filters)):
            int_layer = UNet_segmentation.up_block(int_layer, conv_outputs[i], f)

        int_layer = Conv2D(filters[0] // 2, 3, padding="same", activation="relu")(int_layer)
        outputs = Conv2D(3, (1, 1), padding="same", activation="sigmoid")(int_layer)
        self.model = Model(p0, outputs)

    @staticmethod
    def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        p = MaxPool2D((2, 2), (2, 2))(c)
        return c, p

    @staticmethod
    def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
        us = UpSampling2D((2, 2))(x)
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(us)
        concat = Concatenate()([c, skip])
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
        return c

    @staticmethod
    def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c

    def get_model(self):
        return self.model


class UNet_polyp():
    """ Model from polyp segmentation github
        https://github.com/nikhilroxtomar/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0

        414 435 parameters
    """

    def __init__(self, input_size=(256, 256, 3)):
        num_filters = [16, 32, 48, 64]
        inputs = Input(input_size)

        skip_x = []
        x = inputs

        ## Encoder
        for f in num_filters:
            x = UNet_polyp.conv_block(x, f)
            skip_x.append(x)
            x = MaxPool2D((2, 2))(x)

        ## Bridge
        x = UNet_polyp.conv_block(x, num_filters[-1])

        num_filters.reverse()
        skip_x.reverse()
        ## Decoder
        for i, f in enumerate(num_filters):
            x = UpSampling2D((2, 2))(x)
            xs = skip_x[i]
            x = Concatenate()([x, xs])
            x = UNet_polyp.conv_block(x, f)

        ## Output
        x = Conv2D(3, (1, 1), padding="same")(x)
        x = Activation("sigmoid")(x)

        self.model = Model(inputs, x)

    @staticmethod
    def conv_block(x, num_filters):
        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def get_model(self):
        return self.model
