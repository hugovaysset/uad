import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score


class AccuracyCallback(tf.keras.callbacks.Callback):

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
        accuracy = accuracy_score(y_pred, y_true)
        print(f"Accuracy: {accuracy}")
