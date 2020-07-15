import numpy as np
import tensorflow as tf


class BarycentreSchedule(tf.keras.callbacks.Callback):
    """
    Allows to modify the center of a One-Class neural network after each epoch, replacing it with the
    barycentre (still a tensor) of each predictions on the training set. This could improve convergence of the modelss
    """

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def on_epoch_end(self):
        mean_preds, _, _ = self.model.encoder.predict(self.x_train)
        self.model.CENTER = tf.mean(mean_preds, axis=0)
