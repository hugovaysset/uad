import numpy as np
import tensorflow as tf


class InitializeCenterCallback(tf.keras.callbacks.Callback):
    """
    Allows to modify the center of a One-Class neural network after each epoch, replacing it with the
    barycentre (still a tensor) of each predictions on the training set. This could improve convergence of the modelss
    """

    def __init__(self, x_train):
        """
        Train data from which to compute the center
        """
        self.x_train = x_train

    def on_train_begin(self, logs=None):
        center = self.model.predict(self.x_train)
        self.model.set_CENTER(center)
        print("Center initialized.")
