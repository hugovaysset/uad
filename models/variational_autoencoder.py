# Several Variational Autoencoder implementations:
# - ConvolutionalVAE: fully convolutional with pre-defined architecture
# - VAE: virgin VAE
# - SVDD_VAE : VAE trained using a SVDD loss function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim1, dim2, dim3 = tf.shape(z_mean)[1], tf.shape(z_mean)[2], tf.shape(z_mean)[3]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim1, dim2, dim3))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def conv2d_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=True, activation1="sigmoid", activation2="sigmoid"):
    """Function to add 2 convolutional layers with the parameters passed to it
    activation1: name of the activation function to apply. If none, pass "" (empty string)
    activation2: name of the activation function to apply. If none, pass "" (empty string)
    """
    # first layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
                      kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation1 != "":
        x = layers.Activation('sigmoid')(x)

    # second layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
                      kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation2 !=  "":
        x = layers.Activation('sigmoid')(x)

    return x


class ConvolutionalVAE(Model):
    """
    Implements a Variational Autoencoder with a pre-defined architecture inspired from the
    U-Net models. It contains 3 contraction blocks of (2 convolutions, 1 max pooling) and
    three expansion blocks (1 convolution transpose, 2 convolutions) and sigmoid activation
    function. Well suited for MNIST dataset
    """

    def __init__(self, latent_dim, n_filters=16, k_size=3, batchnorm=False, dropout=0.2, ):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim

        latent_side = int(np.sqrt(latent_dim))

        encoder_inputs = layers.Input(shape=(28, 28, 1), name="encoder_inputs")

        paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0,
                                                         0]])  # shape d x 2 where d is the rank of the tensor and 2 represents "before" and "after"
        x = tf.pad(encoder_inputs, paddings, name="pad")

        # contracting path
        x = self.conv2d_block(x, n_filters * 1, kernel_size=k_size, batchnorm=batchnorm)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)

        x = self.conv2d_block(x, n_filters * 2, kernel_size=k_size, batchnorm=batchnorm)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)

        x = self.conv2d_block(x, n_filters=n_filters * 4, kernel_size=k_size, batchnorm=batchnorm)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)

        z_mean = layers.Conv2D(latent_dim, 1, strides=1, name="z_mean")(x)
        z_log_var = layers.Conv2D(latent_dim, 1, strides=1, name="z_log_var")(x)
        z = Sampling()((z_mean, z_log_var))

        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # Define decoder model.
        latent_inputs = layers.Input(shape=(latent_side, latent_side, latent_dim), name="z_sampling")

        x = layers.Conv2DTranspose(n_filters * 4, (k_size, k_size), strides=(2, 2), padding='same', name="u6")(
            latent_inputs)
        x = layers.Dropout(dropout)(x)
        x = self.conv2d_block(x, n_filters * 4, kernel_size=k_size, batchnorm=batchnorm)

        x = layers.Conv2DTranspose(n_filters * 2, (k_size, k_size), strides=(2, 2), padding='same', name="u7")(x)
        x = layers.Dropout(dropout)(x)
        x = self.conv2d_block(x, n_filters * 2, kernel_size=k_size, batchnorm=batchnorm)

        x = layers.Conv2DTranspose(n_filters * 1, (k_size, k_size), strides=(2, 2), padding='same', name="u8")(x)
        x = layers.Dropout(dropout)(x)
        decoder_outputs = self.conv2d_block(x, 1, kernel_size=k_size, batchnorm=batchnorm)
        crop = tf.image.resize_with_crop_or_pad(decoder_outputs, 28, 28)

        self.decoder = Model(inputs=latent_inputs, outputs=crop, name="decoder")

    def conv2d_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=True):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        c1 = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                           kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            c1 = layers.BatchNormalization()(c1)
        c1 = layers.Activation('sigmoid')(c1)

        # second layer
        c1 = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                           kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            c1 = layers.BatchNormalization()(c1)
        c1 = layers.Activation('sigmoid')(c1)

        return c1

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        # record all the performed operations in order to perform
        # backprop after
        with tf.GradientTape() as tape:
            # predict
            # why not use self.call(data, reconstruction)?
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # compute loss, cannot just use built-in loss
            # because loss not only dependson y_pred, y_true
            # but also on z_mean, z_log_var... Signature mismatch
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss

        # compute gradient of total loss / weights thanks to recorded
        # operations in the with tf.GradientTape() scope
        grads = tape.gradient(total_loss, self.trainable_weights)

        # optimizer defined at compilation
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
        return {"reconstruction_loss": reconstruction_loss}

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def generate_sample(self, n):
        """
        Generate a random sample in the latent space and returns the decoded images
        :param n: number of sample to generate
        :return:
        """
        latent_sample = np.array([tf.random.normal((n, self.latent_dim), mean=0.0, stddev=1.0)])
        latent_sample = np.array(tf.reshape(latent_sample, (n, *self.latent_dim)))
        generated = self.decoder.predict(latent_sample)
        return np.squeeze(generated, axis=-1)


class VAE(Model):
    """
    Variational autoencoder without predefined architecture. Build the encoder and decoder
    using the keras functional API and pass them as arguments to the class to instantiate
    a custom VAE model.
    """
    def __init__(self, encoder, decoder, dims=(28, 28, 1), latent_dim=2, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.dims = dims
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        # record all the performed operations in order to perform
        # backprop after
        with tf.GradientTape() as tape:
            # predict
            # why not use self.call(data, reconstruction)?
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # compute loss, cannot just use built-in loss
            # because loss not only dependson y_pred, y_true
            # but also on z_mean, z_log_var... Signature mismatch
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss

        # compute gradient of total loss / weights thanks to recorded
        # operations in the with tf.GradientTape() scope
        grads = tape.gradient(total_loss, self.trainable_weights)

        # optimizer defined at compilation
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def generate_sample(self, n):
        """
        Generate a random sample in the latent space and returns the decoded images
        :param n: number of sample to generate
        :return:
        """
        latent_sample = np.array([tf.random.normal((n, self.latent_dim), mean=0.0, stddev=1.0)])
        latent_sample = np.array(tf.reshape(latent_sample, (n, self.latent_dim)))
        generated = self.decoder.predict(latent_sample)
        return np.squeeze(generated, axis=-1)


class SVDD_VAE(Model):
    """
    Support Vector Data Description VAE, trained using the associated loss function. The class
    does not come with a pre-defined achitecture. Build the encoder and decoder using the
    functional API and pass them to the constructor.
    """
    def __init__(self, encoder, decoder, dims=(28, 28, 1), latent_dim=2, LAMBDA=1e-6, **kwargs):
        super(SVDD_VAE, self).__init__(**kwargs)
        self.dims = dims
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

        self.CENTER = 1  # find a way to get mean value after the first forward pass
        self.LAMBDA = LAMBDA

    def set_center(self, model, data):
        z_means, z_log_vars, zs = model.encoder.predict(data)
        self.CENTER = tf.reduce_mean(z_means, axis=0)
        print(f"Hypersphere center coordinates: {self.CENTER}")

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(data, reconstruction))
            reconstruction_loss *= 28 * 28

            size = tf.shape(z_mean)[1] + tf.shape(z_mean)[2] + tf.shape(z_mean)[3]
            svdd_loss = tf.cast((1 / size), dtype=tf.float32) * tf.norm(tf.norm(z_mean - self.CENTER, axis=-1, ord="euclidean"), axis=(-2, -1), ord="fro")
            # distance_loss = (1 / size) * tf.cast(tf.norm((z - self.CENTER) ** 2), dtype=tf.double)
            weight_decay = 0
            # for lay in self.encoder.layers:
            #     if lay.trainable_weights != []:
            #         # first norm: compute the Frobenius norm of each kernel -> (n_feature_maps_input, n_fm_output)
            #         # second norm: compute the Frobenius norm on the remaining matrix
            #         weight_decay += tf.cast(tf.norm(tf.norm(lay.trainable_weights[0], axis=(-2, -1), ord="fro") ** 2,
            #                              axis=(-2, -1), ord="fro"), dtype=tf.float64)
            # for lay in self.decoder.layers:
            #     if lay.trainable_weights != []:
            #         weight_decay += tf.cast(tf.norm(tf.norm(lay.trainable_weights[0], axis=(-2, -1), ord="fro") ** 2,
            #                              axis=(-2, -1), ord="fro"), dtype=tf.float64)
            # weight_decay *= self.LAMBDA / 2
            # svdd_loss = weight_decay + distance_loss

            weight = tf.constant(0.8, dtype=tf.float32)
            total_loss = (1 - weight) * reconstruction_loss + weight * tf.cast(svdd_loss, dtype=tf.float32)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "svdd_loss": svdd_loss,
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return z_mean, z_log_var, self.decoder(z)

    def generate_sample(self, n):
        """
        Generate a random sample in the latent space and returns the decoded images
        :param n: number of sample to generate
        :return:
        """
        latent_sample = np.array([tf.random.normal((n, self.latent_dim), mean=0.0, stddev=1.0)])
        latent_sample = np.array(tf.reshape(latent_sample, (n, self.latent_dim)))
        generated = self.decoder.predict(latent_sample)
        return np.squeeze(generated, axis=-1)
