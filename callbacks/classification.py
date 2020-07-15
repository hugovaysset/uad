import numpy as np
import tensorflow as tf
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

    def __init__(self, gt_images, true_labels, criterion="threshold", pix_threshold=0.5, im_threshold=10):
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

    def on_epoch_end(self, epoch, logs={}):
        x_pred = self.model.predict(self.gt_images)
        y_pred = is_anormal(self.gt_images, x_pred, criterion=self.criterion, pix_threshold=self.pix_threshold,
                            im_threshold=self.im_threshold)
        auc = tf.keras.metrics.AUC()
        auc.update_state(self.gt_labels, y_pred)
        print(f"\nAUC = {auc.result()}")

    # def on_train_end(self, epoch, logs={}):

