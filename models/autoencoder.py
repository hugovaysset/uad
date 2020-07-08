# Simple Autoencoders

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class Dense_AE(Model):

    def __init__(self, input_shape, hidden_shape, latent_dim):
        super(Dense_AE, self).__init__()
        self.input_size = input_shape
        self.hidden_shape = hidden_shape
        self.latent_dim = latent_dim

        input_img = layers.Input(shape=(self.input_size,))
        hidden_1 = layers.Dense(self.hidden_shape, activation='relu')(input_img)
        code = layers.Dense(self.latent_dim, activation='relu')(hidden_1)
        hidden_2 = layers.Dense(self.hidden_shape, activation='relu')(code)
        output_img = layers.Dense(self.input_size, activation='sigmoid')(hidden_2)

        self.model = Model(input_img, output_img)


class SimpleConvAE(Model):

    def __init__(self, input_shape, n_filters=4, batchnorm=False, dropout=0.1):
        super(SimpleConvAE, self).__init__()
        self.input_size = input_shape
        self.n_filters = n_filters
        self.batchnorm = batchnorm
        self.dropout = dropout

        input_img = layers.Input(shape=self.input_size)

        # Contracting Path
        c1 = self.conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        p1 = layers.Dropout(dropout)(p1)

        c2 = self.conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        p2 = layers.Dropout(dropout)(p2)

        c5 = self.conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

        u8 = layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c5)
        u8 = layers.Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

        u9 = layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = layers.Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

        output_img = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.model = Model(input_img, output_img)

    def conv2d_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=False):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                          kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('sigmoid')(x)

        # second layer
        x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                          kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('sigmoid')(x)

        return x
