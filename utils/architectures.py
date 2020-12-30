from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf
from enum import Enum


class UNet(Enum):
    """
    Enum class defining different architecture types available
    """
    DEFAULT = 0
    DEFAULT_IMAGENET_EMBEDDING = 1
    DEFAULT_FACENET_EMBEDDING = 2
    RESNET = 3
    POLYP = 4


def build_model(model: UNet, filters=None):
    if model == UNet.DEFAULT:
        print('Using default UNet model')
        if filters is None:
            return UNetDefault(use_embeding=False)
        else:
            return UNetDefault(use_embeding=False, filters=filters)
    elif model == UNet.DEFAULT_IMAGENET_EMBEDDING:
        print('Using default UNet model with imagenet embedding')
        if filters is None:
            return UNetDefault(use_embeding=True)
        else:
            return UNetDefault(use_embeding=True, filters=filters)
    elif model == UNet.DEFAULT_FACENET_EMBEDDING:
        print('Using default UNet model with facenet embedding')
        # TODO: Add facenet embedding
        if filters is None:
            return UNetDefault(use_embeding=True)
        else:
            return UNetDefault(use_embeding=True, filters=filters)
    elif model == UNet.RESNET:
        print('Using UNet Resnet model')
        if filters is None:
            return UNet_resnet()
        else:
            return UNet_resnet(filters=filters)
    else:
        print('Using UNet polyp model')
        return UNet_polyp()


class UNet_resnet:
    """ UNet architecture with resnet blocks
    """
    def __init__(self, input_size=(256, 256, 3), filters=(16, 32, 64, 128, 256)):
        p0 = Input(shape=input_size)
        conv_outputs = []
        first_layer = Conv2D(filters[0], 3, padding='same')(p0)
        int_layer = first_layer
        for i, f in enumerate(filters):
            int_layer, skip = UNet_resnet.down_block(int_layer, f)
            conv_outputs.append(skip)

        int_layer = UNet_resnet.bottleneck(int_layer, filters[-1])

        conv_outputs = list(reversed(conv_outputs))
        reversed_filter = list(reversed(filters))
        for i, f in enumerate(reversed_filter):
            if i + 1 < len(reversed_filter):
                num_filters_next = reversed_filter[i + 1]
            else:
                num_filters_next = f
            int_layer = UNet_resnet.up_block(int_layer, conv_outputs[i], f, num_filters_next)

        # concat. with the first layer
        int_layer = Concatenate()([first_layer, int_layer])
        int_layer = Conv2D(filters[0], 3, padding="same", activation="relu")(int_layer)
        outputs = Conv2D(3, (1, 1), padding="same", activation="sigmoid")(int_layer)
        self.model = Model(p0, outputs)

    @staticmethod
    def down_block(x, num_filters: int = 64):
        # down-sample inputs
        x = Conv2D(num_filters, 3, padding='same', strides=2)(x)

        # inner block
        out = Conv2D(num_filters, 3, padding='same')(x)
        # out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv2D(num_filters, 3, padding='same')(out)

        # merge with the skip connection
        out = Add()([out, x])
        # out = BatchNormalization()(out)
        return Activation('relu')(out), x

    @staticmethod
    def up_block(x, skip, num_filters: int = 64, num_filters_next: int = 64):

        # add U-Net skip connection - before up-sampling
        concat = Concatenate()([x, skip])

        # inner block
        out = Conv2D(num_filters, 3, padding='same')(concat)
        # out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv2D(num_filters, 3, padding='same')(out)

        # merge with the skip connection
        out = Add()([out, x])
        # out = BatchNormalization()(out)
        out = Activation('relu')(out)

        # add U-Net skip connection - before up-sampling
        concat = Concatenate()([out, skip])

        # up-sample
        out = Conv2DTranspose(num_filters_next, 3, padding='same', strides=2)(concat)
        out = Conv2D(num_filters_next, 3, padding='same')(out)
        # out = BatchNormalization()(out)
        return Activation('relu')(out)

    @staticmethod
    def bottleneck(x, filters):
        x = Conv2D(filters, 3, padding='same', name='bottleneck')(x)
        # x = BatchNormalization()(x)
        return Activation('relu')(x)

    def get_model(self):
        return self.model


class UNetDefault:
    """ UNet architecture from following github notebook for image segmentation:
        https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb

        1 963 043 parameters (4,589,283 parameters with embedding from MobileNet in bottleneck layer)
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
            conv_output, int_layer = UNetDefault.down_block(int_layer, f)
            conv_outputs.append(conv_output)

        int_layer = UNetDefault.bottleneck(int_layer, filters[-1])

        if use_embeding:
            int_layer = Concatenate()([int_layer, mn1])

        conv_outputs = list(reversed(conv_outputs))
        for i, f in enumerate(reversed(filters)):
            int_layer = UNetDefault.up_block(int_layer, conv_outputs[i], f)

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

        # Encoder
        for f in num_filters:
            x = UNet_polyp.conv_block(x, f)
            skip_x.append(x)
            x = MaxPool2D((2, 2))(x)

        # Bridge
        x = UNet_polyp.conv_block(x, num_filters[-1])
        num_filters.reverse()
        skip_x.reverse()

        # Decoder
        for i, f in enumerate(num_filters):
            x = UpSampling2D((2, 2))(x)
            xs = skip_x[i]
            x = Concatenate()([x, xs])
            x = UNet_polyp.conv_block(x, f)

        # Output
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
