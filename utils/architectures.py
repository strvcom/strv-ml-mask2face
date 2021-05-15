from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class BaseUNet(ABC):
    """
    Base Interface for UNet
    """

    def __init__(self, model: Model):
        self.model: Model = model

    def get_model(self):
        return self.model

    @staticmethod
    @abstractmethod
    def build_model(input_size: Tuple[int, int, int], filters: Tuple, kernels: Tuple):
        pass


class UNet(Enum):
    """
    Enum class defining different architecture types available
    """
    DEFAULT = 0
    DEFAULT_IMAGENET_EMBEDDING = 1
    RESNET = 3

    def build_model(self, input_size: Tuple[int, int, int], filters: Optional[Tuple] = None,
                    kernels: Optional[Tuple] = None) -> BaseUNet:

        # set default filters
        if filters is None:
            filters = (16, 32, 64, 128, 256)

        # set default kernels
        if kernels is None:
            kernels = list(3 for _ in range(len(filters)))

        # check kernels and filters
        if len(filters) != len(kernels):
            raise Exception('Kernels and filter count has to match.')

        if self == UNet.DEFAULT_IMAGENET_EMBEDDING:
            print('Using default UNet model with imagenet embedding')
            return UNetDefault.build_model(input_size, filters, kernels, use_embedding=True)
        elif self == UNet.RESNET:
            print('Using UNet Resnet model')
            return UNet_resnet.build_model(input_size, filters, kernels)
        print('Using default UNet model')
        return UNetDefault.build_model(input_size, filters, kernels, use_embedding=False)


class UNet_resnet(BaseUNet):
    """
    UNet architecture with resnet blocks
    """

    @staticmethod
    def build_model(input_size: Tuple[int, int, int], filters: Tuple, kernels: Tuple):

        p0 = Input(shape=input_size)
        conv_outputs = []
        first_layer = Conv2D(filters[0], kernels[0], padding='same')(p0)
        int_layer = first_layer
        for i, f in enumerate(filters):
            int_layer, skip = UNet_resnet.down_block(int_layer, f, kernels[i])
            conv_outputs.append(skip)

        int_layer = UNet_resnet.bottleneck(int_layer, filters[-1], kernels[-1])

        conv_outputs = list(reversed(conv_outputs))
        reversed_filter = list(reversed(filters))
        reversed_kernels = list(reversed(kernels))
        for i, f in enumerate(reversed_filter):
            if i + 1 < len(reversed_filter):
                num_filters_next = reversed_filter[i + 1]
                num_kernels_next = reversed_kernels[i + 1]
            else:
                num_filters_next = f
                num_kernels_next = reversed_kernels[i]
            int_layer = UNet_resnet.up_block(int_layer, conv_outputs[i], f, num_filters_next, num_kernels_next)

        # concat. with the first layer
        int_layer = Concatenate()([first_layer, int_layer])
        int_layer = Conv2D(filters[0], kernels[0], padding="same", activation="relu")(int_layer)
        outputs = Conv2D(3, (1, 1), padding="same", activation="sigmoid")(int_layer)
        model = Model(p0, outputs)
        return UNet_resnet(model)

    @staticmethod
    def down_block(x, num_filters: int = 64, kernel: int = 3):
        # down-sample inputs
        x = Conv2D(num_filters, kernel, padding='same', strides=2)(x)

        # inner block
        out = Conv2D(num_filters, kernel, padding='same')(x)
        # out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv2D(num_filters, kernel, padding='same')(out)

        # merge with the skip connection
        out = Add()([out, x])
        # out = BatchNormalization()(out)
        return Activation('relu')(out), x

    @staticmethod
    def up_block(x, skip, num_filters: int = 64, num_filters_next: int = 64, kernel: int = 3):

        # add U-Net skip connection - before up-sampling
        concat = Concatenate()([x, skip])

        # inner block
        out = Conv2D(num_filters, kernel, padding='same')(concat)
        # out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv2D(num_filters, kernel, padding='same')(out)

        # merge with the skip connection
        out = Add()([out, x])
        # out = BatchNormalization()(out)
        out = Activation('relu')(out)

        # add U-Net skip connection - before up-sampling
        concat = Concatenate()([out, skip])

        # up-sample
        # out = UpSampling2D((2, 2))(concat)
        out = Conv2DTranspose(num_filters_next, kernel, padding='same', strides=2)(concat)
        out = Conv2D(num_filters_next, kernel, padding='same')(out)
        # out = BatchNormalization()(out)
        return Activation('relu')(out)

    @staticmethod
    def bottleneck(x, filters, kernel: int = 3):
        x = Conv2D(filters, kernel, padding='same', name='bottleneck')(x)
        # x = BatchNormalization()(x)
        return Activation('relu')(x)


class UNetDefault(BaseUNet):
    """
     UNet architecture from following github notebook for image segmentation:
    https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb
    https://github.com/nikhilroxtomar/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0
    """

    @staticmethod
    def build_model(input_size: Tuple[int, int, int], filters: Tuple, kernels: Tuple, use_embedding: bool = True):

        p0 = Input(input_size)

        if use_embedding:
            mobilenet_model = tf.keras.applications.MobileNetV2(
                input_shape=input_size, include_top=False, weights='imagenet'
            )
            mobilenet_model.trainable = False
            mn1 = mobilenet_model(p0)
            mn1 = Reshape((16, 16, 320))(mn1)

        conv_outputs = []
        int_layer = p0

        for f in filters:
            conv_output, int_layer = UNetDefault.down_block(int_layer, f)
            conv_outputs.append(conv_output)

        int_layer = UNetDefault.bottleneck(int_layer, filters[-1])

        if use_embedding:
            int_layer = Concatenate()([int_layer, mn1])

        conv_outputs = list(reversed(conv_outputs))
        for i, f in enumerate(reversed(filters)):
            int_layer = UNetDefault.up_block(int_layer, conv_outputs[i], f)

        int_layer = Conv2D(filters[0] // 2, 3, padding="same", activation="relu")(int_layer)
        outputs = Conv2D(3, (1, 1), padding="same", activation="sigmoid")(int_layer)
        model = Model(p0, outputs)
        return UNetDefault(model)

    @staticmethod
    def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        # c = BatchNormalization()(c)
        p = MaxPool2D((2, 2), (2, 2))(c)
        return c, p

    @staticmethod
    def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
        us = UpSampling2D((2, 2))(x)
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(us)
        # c = BatchNormalization()(c)
        concat = Concatenate()([c, skip])
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
        # c = BatchNormalization()(c)
        return c

    @staticmethod
    def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        # c = BatchNormalization()(c)
        return c
