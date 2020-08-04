# Several Variational Autoencoder implementations:
# - ConvolutionalVAE: fully convolutional with pre-defined architecture
# - VAE: virgin VAE
# - OC_VAE : VAE trained using a hybrid VAE - SVDD loss function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from uad.decision.oc_vae import anomaly_score
from uad.diagnostic.metrics import binarize_set, is_binary


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim1, dim2, dim3 = tf.shape(z_mean)[1], tf.shape(z_mean)[2], tf.shape(z_mean)[3]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim1, dim2, dim3))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class ConvolutionalVAE(Model):
    """
    Implements a Variational Autoencoder with a pre-defined architecture inspired from the
    U-Net models. It contains 3 contraction blocks of (2 convolutions, 1 max pooling) and
    three expansion blocks (1 convolution transpose, 2 convolutions) and sigmoid activation
    function. Well suited for MNIST dataset
    """

    def __init__(self, latent_dim, n_filters=16, k_size=3, batchnorm=False, dropout=0.2, activation_func="relu"):
        super(ConvolutionalVAE, self).__init__()
        self.latent_dim = latent_dim

        latent_side = 4

        encoder_inputs = layers.Input(shape=(28, 28, 1), name="encoder_inputs")

        paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0,
                                                         0]])  # shape d x 2 where d is the rank of the tensor and 2 represents "before" and "after"
        x = tf.pad(encoder_inputs, paddings, name="pad")

        # contracting path
        x = self.conv2d_block(x, n_filters * 1, kernel_size=k_size, batchnorm=batchnorm, activation=activation_func)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)

        x = self.conv2d_block(x, n_filters * 2, kernel_size=k_size, batchnorm=batchnorm, activation=activation_func)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)

        x = self.conv2d_block(x, n_filters=n_filters * 4, kernel_size=k_size, batchnorm=batchnorm,
                              activation=activation_func)
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
        x = self.conv2d_block(x, n_filters * 4, kernel_size=k_size, batchnorm=batchnorm, activation=activation_func)

        x = layers.Conv2DTranspose(n_filters * 2, (k_size, k_size), strides=(2, 2), padding='same', name="u7")(x)
        x = layers.Dropout(dropout)(x)
        x = self.conv2d_block(x, n_filters * 2, kernel_size=k_size, batchnorm=batchnorm, activation=activation_func)

        x = layers.Conv2DTranspose(n_filters * 1, (k_size, k_size), strides=(2, 2), padding='same', name="u8")(x)
        x = layers.Dropout(dropout)(x)
        decoder_outputs = self.conv2d_block(x, 1, kernel_size=k_size, batchnorm=batchnorm, activation=activation_func)
        crop = tf.image.resize_with_crop_or_pad(decoder_outputs, 28, 28)

        self.decoder = Model(inputs=latent_inputs, outputs=crop, name="decoder")

    def conv2d_block(self, input_tensor, n_filters, kernel_size=(3, 1), batchnorm=True,
                     activation="relu"):
        """Function to add 2 convolutional layers with the parameters passed to it
        activation1: name of the activation function to apply. If none, pass "" (empty string)
        activation2: name of the activation function to apply. If none, pass "" (empty string)
        """
        # first layer
        x = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                          kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        if activation == "relu" or activation == "sigmoid" or activation == "linear":
            x = layers.Activation(activation)(x)
        elif activation == "leaky_relu":
            x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        else:
            raise NotImplementedError("activation function should be given by a valid string of leaky_relu")

        # second layer
        x = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                          kernel_initializer='he_normal', padding='same')(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        if activation == "relu" or activation == "sigmoid" or activation == "linear":
            x = layers.Activation(activation)(x)
        elif activation == "leaky_relu":
            x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        else:
            raise NotImplementedError("activation function should be given by a valid string of leaky_relu")

        return x

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

    def __init__(self, encoder, decoder, dims=(28, 28, 1), reconstruction_loss="mse", BETA=1, **kwargs):
        """
        :param encoder:
        :param decoder:
        :param dims:
        :param reconstruction_loss: name of the reconstruction loss to use (can be "xent" for MNIST or "mse" for real
        images
        """
        super(VAE, self).__init__(**kwargs)
        self.dims = dims
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss = reconstruction_loss

        self.BETA = BETA

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:

            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            if self.reconstruction_loss == "xent":
                reconstruction_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(data, reconstruction)
                )
                reconstruction_loss *= self.dims[0] * self.dims[1]
            elif self.reconstruction_loss == "mse":
                reconstruction_loss = tf.reduce_mean(tf.reduce_sum((reconstruction - data) ** 2, axis=(1, 2, 3)))
            else:
                raise NotImplementedError("Reconstruction loss should be either xent or mse")
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + self.BETA * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
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

            if self.reconstruction_loss == "xent":
                reconstruction_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(data, reconstruction)
                )
                reconstruction_loss *= self.dims[0] * self.dims[1]
            elif self.reconstruction_loss == "mse":
                reconstruction_loss = tf.reduce_mean(tf.reduce_sum((reconstruction - data) ** 2, axis=(1, 2, 3)),
                                                     axis=0)
            else:
                raise NotImplementedError("Reconstruction loss should be either xent or mse")
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

    def score_samples_iterator(self, dataset_iterator):
        """
        Returns the anomaly scores for data (name of the method inspired from the sklearn
        interface) when data is given in an iterator
        :param dataset_iterator: image or batch of images
        :param decision_function: can be either "distance" to predict anomalies based on their distance to the model's center
        (in an SVDD manner) or "reconstruction" to predict anomalies based on the reconstruction error between the input
        image and the reconstruction (using MSE, in a VAE manner).
        Return: scores in the batch format
        """
        scores = []
        for i in range(len(dataset_iterator)):
            _, (ims, labs) = dataset_iterator[i]
            if (i + 1) % 50 == 0:
                print(f"making predictions on batch {i + 1}...")
            predictions = self.predict(ims)
            y_scores = tf.math.sqrt(tf.reduce_sum((predictions - ims) ** 2, axis=(
            1, 2, 3)))  # implementer differentes fonctions de decision ensuite
            scores.append(y_scores)
        return np.array(scores)

    def compute_ROC_iterator(self, dataset_iterator, interest_digit=0):
        """
        :param dataset_iterator:
        :param decision_function:
        :param batch:
        :param interest_digit:
        :return:
        """
        labels = []
        for i in range(len(dataset_iterator)):
            _, (ims, y_true) = dataset_iterator[i]
            labels.append(y_true.squeeze(-1))

        y_trues = np.array(labels).flatten()
        y_scores = self.score_samples_iterator(dataset_iterator).flatten()

        if not is_binary(y_trues):
            y_true_bin = binarize_set(y_trues, interest=interest_digit)
        else:
            y_true_bin = y_trues

        fpr, tpr, thresholds = roc_curve(y_true_bin, y_scores, pos_label=1)

        return fpr, tpr, thresholds

    def plot_scores_distrib(self, dataset_iterator, interest_class=0):
        """
        Plot the distribution of anomaly scores computed on dataset_iterator, for
        the normal class and for the anormal class
        :param dataset_iterator:
        :param interest_class:
        :return:
        """
        labs = []
        for i in range(len(dataset_iterator)):
            _, (ims, lab) = dataset_iterator[i]
            labs.append(lab)
        labs = np.array(labs).squeeze(-1).flatten()
        sc = self.score_samples_iterator(dataset_iterator).flatten()

        scores_nominal = sc[labs == interest_class]
        scores_anormal = sc[labs != interest_class]

        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        axes[0].hist(scores_nominal)
        axes[0].set_title("anomaly scores for nominal class")
        axes[1].hist(scores_anormal)
        axes[1].set_title("anomaly scores for anormal class")

        return fig, axes


class OC_VAE(Model):
    """
    Hybrid network between a variational autoencoder and a deep-SVDD network. It is trained using a (possibly) weighted
    sum of reconstruction error, KL-divergence and centripetal loss. The weights regularization is not included because
    the reconstruction loss should prevent from the sphere collapse phenomenon.
    """

    def __init__(self, encoder, decoder, input_dims=(28, 28, 1), latent_dims=(4, 4, 16),
                 LAMBDAS=(0.33, 0.33), reconstruction_loss="mse", **kwargs):
        super(OC_VAE, self).__init__(**kwargs)
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.latent_dim = latent_dims[-1]
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss = reconstruction_loss
        self.CENTER = tf.Variable(initial_value=np.ones(latent_dims),
                                  dtype=tf.float32)  # center of the same size as output
        self.LAMBDAS = LAMBDAS

    def train_step(self, data):
        if isinstance(data, tuple):
            data, denoised_data = data
        else:
            denoised_data = None

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            if self.reconstruction_loss == "xent":
                reconstruction_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(data, reconstruction)
                )
                reconstruction_loss *= self.input_dims[0] * self.input_dims[1]
            elif self.reconstruction_loss == "mse":
                if denoised_data is not None:
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum((reconstruction - denoised_data) ** 2, axis=(1, 2, 3)), axis=0)
                else:
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum((reconstruction - data) ** 2, axis=(1, 2, 3)), axis=0)
            else:
                raise NotImplementedError("Reconstruction loss should be either xent or mse")

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5

            # return a batch loss or a scalar (taking the mean loss) ?
            centripetal_loss = tf.math.sqrt(tf.reduce_sum((z_mean - self.CENTER) ** 2))

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
            data, denoised_data = data
        else:
            denoised_data = None

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            if self.reconstruction_loss == "xent":
                reconstruction_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(data, reconstruction)
                )
                reconstruction_loss *= self.input_dims[0] * self.input_dims[1]
            elif self.reconstruction_loss == "mse":
                if denoised_data is not None:
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum((reconstruction - denoised_data) ** 2, axis=(1, 2, 3)), axis=0)
                else:
                    reconstruction_loss = tf.reduce_mean(
                        tf.reduce_sum((reconstruction - data) ** 2, axis=(1, 2, 3)), axis=0)
            else:
                raise NotImplementedError("Reconstruction loss should be either xent or mse")

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5

            centripetal_loss = tf.math.sqrt(tf.reduce_sum((z_mean - self.CENTER) ** 2))

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

    def score_samples_iterator(self, dataset_iterator, decision_function="distance", batch=True):
        """
        Returns the anomaly scores for data (name of the method inspired from the sklearn
        interface) when data is given in an iterator
        :param dataset_iterator: image or batch of images
        :param decision_function: can be either "distance" to predict anomalies based on their distance to the model's center
        (in an SVDD manner) or "reconstruction" to predict anomalies based on the reconstruction error between the input
        image and the reconstruction (using MSE, in a VAE manner).
        Return: scores in the batch format
        """
        scores = []
        for i in range(len(dataset_iterator)):  # itere a l'infini???
            _, (ims, labs) = dataset_iterator[i]
            if (i + 1) % 50 == 0:
                print(f"making predictions on batch {i + 1}...")
            if decision_function == "distance":
                predictions = self.encoder.predict(ims)
            elif decision_function == "reconstruction":
                predictions = self.predict(ims)
            else:
                raise NotImplementedError("decision function should be either 'distance' or 'reconstruction'")
            y_scores = anomaly_score(self, dataset_iterator[i], decision_func=decision_function, batch=batch)
            scores.append(y_scores)
        return np.array(scores)

    def is_anormal(self, data, im_threshold=0):
        scores = self.score_samples(data)
        return binarize_set(scores > im_threshold)

    def compute_ROC(self, y_true, y_score):
        return roc_curve(y_true, y_score, pos_label=0)

    def compute_ROC_iterator(self, dataset_iterator, decision_function="distance", batch=True, interest_digit=7):
        """
        :param dataset_iterator:
        :param decision_function:
        :param batch:
        :param interest_digit:
        :return:
        """
        labels = []
        for i in range(len(dataset_iterator)):
            _, (ims, y_true) = dataset_iterator[i]
            labels.append(y_true.squeeze(-1))
        y_trues = np.array(labels).flatten()

        y_scores = self.score_samples_iterator(dataset_iterator, decision_function=decision_function).flatten()

        if not is_binary(y_trues):
            y_true_bin = binarize_set(y_trues, interest=interest_digit)
        else:
            y_true_bin = y_trues

        fpr, tpr, thresholds = roc_curve(y_true_bin, y_scores)

        return fpr, tpr, thresholds

    def compute_AUC(self, fprs, tprs):
        return auc(fprs, tprs)

    def plot_scores_distrib(self, dataset_iterator, decision_function="distance", batch=True, interest_class=7):
        """
        Plot the distribution of anomaly scores computed on dataset_iterator, for
        the normal class and for the anormal class
        :param dataset_iterator:
        :param interest_class:
        :return:
        """
        labs = []
        for i in range(len(dataset_iterator)):
            _, (ims, lab) = dataset_iterator[i]
            labs.append(lab)
        labs = np.array(labs).squeeze(-1).flatten()

        sc = self.score_samples_iterator(dataset_iterator, decision_function=decision_function, batch=batch).flatten()

        scores_nominal = sc[labs == interest_class]
        scores_anormal = sc[labs != interest_class]

        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        axes[0].hist(scores_nominal)
        axes[0].set_title("anomaly scores for nominal class")
        axes[1].hist(scores_anormal)
        axes[1].set_title("anomaly scores for anormal class")

        return fig, axes
