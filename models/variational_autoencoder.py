# Several Variational Autoencoder implementations:
# - ConvolutionalVAE: fully convolutional with pre-defined architecture
# - VAE: virgin VAE
# - OC_VAE : VAE trained using a hybrid VAE - SVDD loss function

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from uad.decision.oc_vae import anomaly_score
from uad.diagnostic.metrics import binarize_set


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim1, dim2, dim3 = tf.shape(z_mean)[1], tf.shape(z_mean)[2], tf.shape(z_mean)[3]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim1, dim2, dim3))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, activation1="sigmoid",
                 activation2="sigmoid"):
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
        x = layers.Activation(activation1)(x)

    # second layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
                      kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation2 != "":
        x = layers.Activation(activation2)(x)

    return x


class ConvolutionalVAE(Model):
    """
    Implements a Variational Autoencoder with a pre-defined architecture inspired from the
    U-Net models. It contains 3 contraction blocks of (2 convolutions, 1 max pooling) and
    three expansion blocks (1 convolution transpose, 2 convolutions) and sigmoid activation
    function. Well suited for MNIST dataset
    """

    def __init__(self, latent_dim, n_filters=16, k_size=3, batchnorm=False, dropout=0.2):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim

        latent_side = 4

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
        """
        We give (x_val, x_val) as validation data and not (x_val, y_val), which could
        be pratictal to directly compute accuracy and reconstruction loss, but is
        inconsistent with train_step arguments.
        :param data:
        :return:
        """
        x_val = data[0]
        z_mean, z_log_var, z = self.encoder(x_val)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x_val, reconstruction))
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


class OC_VAE(Model):
    """
    Hybrid network between a variational autoencoder and a deep-SVDD network. It is trained using a (possibly) weighted
    sum of reconstruction error, KL-divergence and centripetal loss. The weights regularization is not included because
    the reconstruction loss should prevent from the sphere collapse phenomenon.
    """

    def __init__(self, encoder, decoder, dims=(28, 28, 1), latent_dims=(4, 4, 16), LAMBDAS=(0.33, 0.33), **kwargs):
        super(OC_VAE, self).__init__(**kwargs)
        self.dims = dims
        self.latent_dims = latent_dims
        self.latent_dim = latent_dims[-1]
        self.encoder = encoder
        self.decoder = decoder

        self.CENTER = tf.Variable(initial_value=np.ones(latent_dims),
                                  dtype=tf.float32)  # center of the same size as output
        self.LAMBDAS = LAMBDAS

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, reconstruction)  # change to MSE for other images
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            centripetal_loss = tf.math.sqrt(tf.reduce_sum(z_mean - self.CENTER) ** 2)
            total_loss = (1 - (self.LAMBDAS[0] + self.LAMBDAS[1])) * reconstruction_loss + self.LAMBDAS[0] * kl_loss + \
                         self.LAMBDAS[1] * centripetal_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "centripetal_loss": centripetal_loss,
            "kl_loss": kl_loss,
        }

    def set_center(self, new_center):
        self.CENTER = new_center
        tf.print(f"Hypersphere center coordinates: {self.CENTER}")

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
            centripetal_loss = tf.math.sqrt(tf.reduce_sum(z_mean - self.CENTER) ** 2)
            total_loss = (1 - (self.LAMBDAS[0] + self.LAMBDAS[1])) * reconstruction_loss + self.LAMBDAS[0] * kl_loss + \
                         self.LAMBDAS[1] * centripetal_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "centripetal_loss": centripetal_loss,
            "kl_loss": kl_loss,
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def distance_to_center(self, data):
        z_mean, z_log_var, z = self.encoder.predict(data)
        return tf.math.sqrt(tf.reduce_sum(z_mean - self.CENTER) ** 2)

    def score_samples(self, data, decision_function="distance", batch=True):
        """
        Returns the anomaly scores for data (name of the method inspired from the sklearn
        interface)
        :param data: image or batch of images
        :param decision_func: can be either "distance" to predict anomalies based on their distance to the model's center
        (in an SVDD manner) or "reconstruction" to predict anomalies based on the reconstruction error between the input
        image and the reconstruction (using MSE, in a VAE manner).
        :param batch: True if the given data is a batch
        :return: anomaly scores, in a numpy vector or a single value depending on the data type
        """
        return - anomaly_score(self, data, decision_func=decision_function, batch=batch).numpy()

    def is_anormal(self, data, im_threshold=0):
        scores = self.score_samples(data)
        return binarize_set(scores > im_threshold)

    def compute_ROC(self, y_true, y_score):
        return roc_curve(y_true, y_score, pos_label=0)

    def compute_AUC(self, fprs, tprs):
        return auc(fprs, tprs)
