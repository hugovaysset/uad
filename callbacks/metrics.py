import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from uad.decision.reconstruction import is_anormal

class AUCCallback(tf.keras.callbacks.Callback):
    """
    Callback used for any model that contains a "is_anormal" method (e.g. A VAE predicts images from images. The
    associated is_anormal method predicts labels (0 for normal, 1 for anormal) from those images and of course ground
    truth images.
    - After each training epoch: computes AUC score to measure the model performance
    - In the end of the training step : plots the ROC curve of the model (to do: store the ROC curve every two or three
    steps and plot them all in the end)
    """

    def __init__(self, gt_images, true_labels, criterion="threshold", pix_threshold=0.5, im_threshold=10, prefix="val"):
        """
        :param gt_images: reference images (validation set)
        :param true_labels: array containing the true labels binarized (i.e. 0 for normal class and 1 for all other
        class which is considered as abnormal)
        """
        self.gt_images = gt_images
        self.gt_labels = true_labels
        self.criterion = criterion
        self.pix_threshold = pix_threshold
        self.im_threshold = im_threshold
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs={}):
        x_pred = self.model.predict(self.gt_images)
        y_pred = is_anormal(self.gt_images, x_pred, criterion=self.criterion, pix_threshold=self.pix_threshold,
                            im_threshold=self.im_threshold)
        auc = tf.keras.metrics.AUC()
        auc.update_state(self.gt_labels, y_pred)
        res = auc.result()
        logs[f"{self.prefix}_AUC"] = res
        print(f"\nAUC = {res}")

    # def on_train_end(self, epoch, logs={}):


class PrecisionRecallCallback(tf.keras.callbacks.Callback):
    """
    Computes precision and recall during the training step, in the end of each epoch. Takes as arguments the input data x, y on which to
    compute accuracy. To get the accuracy score on both the training and validation data, create two instances of the model and
    give them as callbacks. To differentiate each instance, you can give a prefix to the metric name in the history object
    returned by the fit() method. A custom callback is dedicated to compute the accuracy since for a VAE, the prediction
    is on x_prediction so the method could not fit the standard signature of all Keras built-in metrics (y_pred, y_train).
    The reconstruction loss is computed during training between (x_true, x_pred) so a custom callback is not dedicated to that.
    """

    def __init__(self, gt_images, true_labels, criterion="l2", pix_threshold=0.5, im_threshold=10, prefix="val"):
        """
        :param gt_images: reference images (validation set)
        :param true_labels: array containing the true labels binarized (i.e. 0 for normal class and 1 for all other
        class which is considered as abnormal)
        """
        self.gt_images = gt_images
        self.gt_labels = true_labels
        self.criterion = criterion
        self.pix_threshold = pix_threshold
        self.im_threshold = im_threshold
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs={}):
        x_pred = self.model.predict(self.gt_images)
        y_pred = is_anormal(self.gt_images, x_pred, criterion=self.criterion, pix_threshold=self.pix_threshold,
                            im_threshold=self.im_threshold)
        precision = tf.keras.metrics.Precision()
        recall = tf.keras.metrics.Recall()
        precision.update_state(self.gt_labels, y_pred)
        recall.update_state(self.gt_labels, y_pred)
        precision_res = precision.result().numpy()
        recall_res = recall.result().numpy()
        logs[f"{self.prefix}_precision"] = precision_res * 100
        logs[f"{self.prefix}_recall"] = recall_res * 100
        print(f" - {self.prefix}_accuracy: {precision_res * 100}%")
        print(f" - {self.prefix}_recall: {recall_res * 100}%")