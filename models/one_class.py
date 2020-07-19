import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import Constraint

from uad.decision.deep_svdd import anomaly_score_from_predictions, is_anormal


class DeepSVDD(Model):
    """
    Support Vector Data Description neural network. Trained on original data and learns a dense embedding while
    trained on the objective function.
    """

    def __init__(self, n_filters=(8, 4), dense_shape=32, LAMBDA=1e-6, **kwargs):
        """
        If inputs is None and outputs is None: builds a DeepSVDD network with a LeNet architecture as used in Ruff 2018
        Else give input and outputs to build a model via subclassing
        :param n_filters: # filters for each convolution. Tuple length must match the number of blocks
        :param dims: input shape
        :param dense_shape: number of units of the final dense layer
        :param LAMBDA:
        :param kwargs:
        """
        super(DeepSVDD, self).__init__(**kwargs)
        self.CENTER = tf.Variable(initial_value=np.ones(dense_shape))  # center of the same size as output
        self.LAMBDA = LAMBDA

        self.c1 = layers.Conv2D(filters=n_filters[0], kernel_size=5, strides=(1, 1), kernel_regularizer=l2(self.LAMBDA),
                                bias_regularizer=l2(self.LAMBDA), padding="same", name=f"conv_1")(self.inputs)
        self.a1 = layers.LeakyReLU(alpha=0.1, name=f"activation_1")(self.c1)
        self.mp1 = layers.MaxPooling2D((2, 2), name=f"max_pooling_1")(self.a1)
        self.c2 = layers.Conv2D(filters=n_filters[1], kernel_size=5, strides=(1, 1), kernel_regularizer=l2(self.LAMBDA),
                                bias_regularizer=l2(self.LAMBDA), padding="same", name=f"conv_2")(
            self.mp1)
        self.a2 = layers.LeakyReLU(alpha=0.1, name=f"activation_2")(self.c2)
        self.mp2 = layers.MaxPooling2D((2, 2), name=f"max_pooling_2")(self.a2)
        self.outputs = layers.Dense(dense_shape, kernel_regularizer=l2(self.LAMBDA), bias_regularizer=l2(self.LAMBDA), )

    def set_center(self, new_center):
        self.CENTER = new_center
        print(f"Hypersphere center coordinates: {self.CENTER}")

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            predictions = self.call(data)
            centripetal_loss = tf.reduce_mean(tf.norm(predictions - self.CENTER) ** 2)
        grads = tape.gradient(centripetal_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "centripetal_loss": centripetal_loss,
        }

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.a1(x)
        x = self.mp1(x)

        x = self.c2(x)
        x = self.a2(x)
        return self.outputs(x)

    def score_samples(self, data):
        """
        Returns the anomaly scores for data (name of the method inspired from the sklearn
        interface)
        :param data: image or batch of images
        :return: anomaly scores
        """
        return anomaly_score_from_predictions(self, self.call(data))

    def is_anormal(self, data, threshold=10):
        """
        Predict if each sample of data is normal or anormal
        :param data:
        :param threshold:
        :return:
        """
        return is_anormal(self, self.predict(data), threshold=threshold)


class SVDDVAE(Model):
    """
    Support Vector Data Description VAE, trained using the associated loss function. The class
    does not come with a pre-defined achitecture. Build the encoder and decoder using the
    functional API and pass them to the constructor.
    """

    def __init__(self, encoder, decoder, dims=(28, 28, 1), latent_dim=2, LAMBDA=1e-6, **kwargs):
        super(SVDDVAE, self).__init__(**kwargs)
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
            svdd_loss = tf.cast((1 / size), dtype=tf.float32) * tf.norm(
                tf.norm(z_mean - self.CENTER, axis=-1, ord="euclidean"), axis=(-2, -1), ord="fro")
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


class CenterAround(Constraint):
    """Constrains weight tensors to be centered around `ref_value`."""

    def __init__(self, ref_value):
        self.ref_value = ref_value

    def __call__(self, w):
        mean = tf.reduce_mean(w)
        return w - mean + self.ref_value

    def get_config(self):
        return {'ref_value': self.ref_value}
