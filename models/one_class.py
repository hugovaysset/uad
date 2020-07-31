import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from uad.decision.deep_svdd import anomaly_score_from_predictions, anomaly_score_from_images
from uad.diagnostic.metrics import binarize_set, is_binary


class DeepSVDD(Model):
    """
    Support Vector Data Description neural network. Trained on original data and learns a dense embedding while
    trained on the objective function.
    """

    def __init__(self, model, dense_shape=32, LAMBDA=1e-6, **kwargs):
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

    def score_samples_iterator(self, dataset_iterator):
        """
        Compute scores (mse) between images and predictions.
        :param dataset_iterator: iterator of MultiChannelIterator type
        Return: scores in the batch format
        """
        scores = []
        for i in range(len(dataset_iterator)):  # itere a l'infini???
            _, (ims, labs) = dataset_iterator[i]
            if (i + 1) % 50 == 0:
                print(f"making predictions on batch {i + 1}...")
            predictions = self.predict(ims)
            y_scores = np.sum((predictions - self.CENTER) ** 2, axis=(1, 2, 3))
            scores.append(y_scores)

        return np.array(scores)

    def is_anormal(self, data, im_threshold=0):
        predictions = self.predict(data)
        return binarize_set(np.sum((predictions - self.CENTER) ** 2, axis=-1) > im_threshold)

    def compute_AUC(self, fprs, tprs):
        return auc(fprs, tprs)

    def compute_ROC(self, y_true, y_score):
        return roc_curve(y_true, y_score)

    def compute_ROC_iterator(self, dataset_iterator, interest_digit=7):
        """
        :param dataset_iterator: expected to be in the format given by
        MultiChannelIterator with (bx, by) == dataset_iterator[0] and then
        (images_x, labels_x) == bx
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

        fpr, tpr, thresholds = roc_curve(y_true_bin, y_scores)

        return fpr, tpr, thresholds

    def plot_scores_distrib(self, dataset_iterator, interest_class=7):
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
