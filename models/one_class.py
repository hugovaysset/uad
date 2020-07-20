import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import Constraint
from uad.decision.deep_svdd import anomaly_score_from_predictions, anomaly_score_from_images, is_anormal
from uad.diagnostic.metrics import binarize_set


class DeepSVDD(Model):
    """
    Support Vector Data Description neural network. Trained on original data and learns a dense embedding while
    trained on the objective function.
    """

    def __init__(self, model, n_filters=(8, 4), dense_shape=32, LAMBDA=1e-6, **kwargs):
        """
        If inputs is None and outputs is None: builds a DeepSVDD network with a LeNet architecture as used in Ruff 2018
        Else give input and outputs to build a model via subclassing
        :param model: either an iterable of layers or a keras.Model
        :param n_filters: # filters for each convolution. Tuple length must match the number of blocks
        :param dims: input shape
        :param dense_shape: number of units of the final dense layer
        :param LAMBDA: factor in front of the weight decay loss
        :param kwargs:
        """
        super(DeepSVDD, self).__init__(**kwargs)
        self.CENTER = tf.Variable(initial_value=np.ones(dense_shape),
                                  dtype=tf.float32)  # center of the same size as output
        self.RADIUS = 0
        self.LAMBDA = tf.constant(LAMBDA, dtype=tf.float32)
        self.model = model

    def set_center(self, new_center):
        self.CENTER = new_center
        tf.print(f"Hypersphere center coordinates: {self.CENTER}")

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            predictions = self(data)
            distances_to_center = tf.norm(predictions - self.CENTER, axis=-1)
            self.RADIUS = tf.reduce_max(distances_to_center)
            centripetal_loss = tf.reduce_mean(distances_to_center ** 2)
            weight_decay = tf.math.reduce_sum(self.losses)
            total_loss = centripetal_loss + self.LAMBDA * weight_decay
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "total_loss": total_loss,
            "centripetal_loss": centripetal_loss,
            "weight_decay": weight_decay
        }

    def call(self, inputs):
        if callable(self.model):
            return self.model(inputs)
        else:
            x = inputs
            for lay in self.model:
                x = lay(x)
            return x

    def score_samples(self, data):
        """
        Returns the anomaly scores for data (name of the method inspired from the sklearn
        interface)
        :param data: image or batch of images
        :return: anomaly scores
        """
        return (anomaly_score_from_images(self, data)).numpy()

    def is_anormal(self, data, im_threshold=0):
        predictions = self.predict(data)
        return binarize_set(np.sum((predictions - self.CENTER) ** 2, axis=-1) > im_threshold)

    def compute_AUC(self, fprs, tprs):
        return auc(fprs, tprs)

    def compute_ROC(self, y_true, y_score):
        return roc_curve(y_true, y_score)

    @staticmethod
    def evaluate_on_all(archi, x_train, x_test, y_test, n_classes=10, epochs=30, **params):
        """
        Compute AUC score for the training on each class of the dataset
        """
        auc_scores = []
        for k in range(n_classes):
            print(f"Digit {k}, # Training examples: {x_train[k].shape[0]}")
            model = DeepSVDD(archi, LAMBDA=1e-5)
            model.compile(optimizer=tf.keras.optimizers.Adam())
            model.fit(x_train[k], epochs=epochs, batch_size=128)
            y_score = model.score_samples(x_test)
            fpr, tpr, _ = model.compute_ROC(y_test, y_score)
            auc = model.compute_AUC(fpr, tpr)
            auc_scores.append(auc)

        return np.array(auc_scores)


def get_ruff_model():
    n_filters = (16, 32, 64)
    dense_shape = 32
    LAMBDA = 1e-6

    inputs = tf.keras.Input(shape=(28, 28, 1))
    paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0,
                                                     0]])  # shape d x 2 where d is the rank of the tensor and 2 represents "before" and "after"
    x = tf.pad(inputs, paddings, name="pad")

    c1 = layers.Conv2D(filters=n_filters[0], kernel_size=5, strides=(1, 1), kernel_regularizer=l2(LAMBDA),
                       padding="same", name=f"conv_1")(x)
    b1 = layers.BatchNormalization()(c1)
    a1 = layers.LeakyReLU(alpha=0.1, name=f"activation_1")(c1)
    mp1 = layers.MaxPooling2D((2, 2), name=f"max_pooling_1")(a1)

    c2 = layers.Conv2D(filters=n_filters[1], kernel_size=5, strides=(1, 1), kernel_regularizer=l2(LAMBDA),
                       padding="same", name=f"conv_2")(mp1)
    b2 = layers.BatchNormalization()(c2)
    a2 = layers.LeakyReLU(alpha=0.1, name=f"activation_2")(b2)
    mp2 = layers.MaxPooling2D((2, 2), name=f"max_pooling_2")(a2)

    c3 = layers.Conv2D(filters=n_filters[2], kernel_size=5, strides=(1, 1), kernel_regularizer=l2(LAMBDA),
                       padding="same", name=f"conv_3")(mp2)
    b3 = layers.BatchNormalization()(c3)
    a3 = layers.LeakyReLU(alpha=0.1, name=f"activation_3")(b3)
    mp3 = layers.MaxPooling2D((2, 2), name=f"max_pooling_3")(a3)
    f3 = layers.Flatten()(mp3)

    d4 = layers.Dense(64, kernel_regularizer=l2(LAMBDA))(f3)
    outputs = layers.Dense(32, kernel_regularizer=l2(LAMBDA))(d4)

    return tf.keras.Model(inputs, outputs)


def conv2d_block(input_tensor, n_filters, kernel_size=3, LAMBDA=1e-6):
    """Function to add 2 convolutional layers with the parameters passed to it
    activation1: name of the activation function to apply. If none, pass "" (empty string)
    activation2: name of the activation function to apply. If none, pass "" (empty string)
    """
    # first layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                      kernel_regularizer=l2(LAMBDA), bias_regularizer=l2(LAMBDA),
                      kernel_initializer='he_normal', padding='same')(input_tensor)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # second layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                      kernel_regularizer=l2(LAMBDA), bias_regularizer=l2(LAMBDA),
                      kernel_initializer='he_normal', padding='same')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    return x


def get_model():
    n_filters = 8
    dense_shape = 32
    LAMBDA = 1
    dropout = None
    k_size = 3

    inputs = layers.Input(shape=(28, 28, 1), name="inputs")
    paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0,
                                                     0]])  # shape d x 2 where d is the rank of the tensor and 2 represents "before" and "after"
    x = tf.pad(inputs, paddings, name="pad")

    x = conv2d_block(x, n_filters * 1, kernel_size=k_size, LAMBDA=LAMBDA)
    x = layers.MaxPooling2D((2, 2))(x)
    # x = layers.Dropout(dropout)(x)

    x = conv2d_block(x, n_filters * 2, kernel_size=k_size, LAMBDA=LAMBDA)
    x = layers.MaxPooling2D((2, 2))(x)
    # x = layers.Dropout(dropout)(x)

    x = conv2d_block(x, n_filters=n_filters * 4, kernel_size=k_size, LAMBDA=LAMBDA)
    x = layers.MaxPooling2D((2, 2))(x)
    # x = layers.Dropout(dropout)(x)

    x = conv2d_block(x, 1, kernel_size=k_size, LAMBDA=LAMBDA)

    outputs = layers.Flatten()(x)

    return tf.keras.Model(inputs, outputs)
