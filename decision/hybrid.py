import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.svm import OneClassSVM
from uad.models.variational_autoencoder import ConvolutionalVAE


class OCHybrid(BaseEstimator, OutlierMixin):
    """
    Hybrid model that takes a pretrained encoder (typically the encoder part of a VAE) and predicts
    anomalies from the encoded signal. Warning : the following the convention is adopted 1 for
    anomalies and 0 for normal. This is opposite to the sklearn convention
    """

    def __init__(self, encoder=None, params={}):
        """
        :param encoder: model that implements predict(batch) --> z_means, z_log_vars, zs
        :param params: additional parameters for the OneClassSVM part
        """
        self.encoder = encoder
        self.svm = OneClassSVM(**params)
        self.svm_fitted_ = False

    def binarize(self, Z):
        """
        Convert from sklearn convention {anormal: -1, normal: 1} to our convention {anormal:1,
        normal:0}
        :param Z:
        :return:
        """
        l = []
        for elt in Z:
            if elt == -1:
                l.append(1)
            else:
                l.append(0)
        return np.array(l)

    def encode(self, x):
        """
        Encode the given images and returns a batch of vectors
        :param x: original input
        :return: flatten encoded input
        """
        if len(x.shape) != 3 and len(x.shape) != 4:
            raise NotImplementedError("Input tensor should be either (None, x, y, z) or (None, x, y)")

        z_means, z_log_vars, zs = self.encoder.predict(x)

        if len(z_means.shape) == 1:  # flat vector of real numbers
            batch = z_means.shape[0]
            z_m = np.reshape(z_means, (batch, 1))
        else:
            batch, flatten = z_means.shape[0], np.prod(z_means.shape[1:])
            z_m = np.reshape(z_means, (batch, flatten))

        return z_m

    def fit(self, x_train):
        """
        encode the train set into the appropriate format for the OC-
        SVM to be trained
        :param x_train: original train dataset. x_train should be either (None, x, y, z)
        (batch of 3-tensor) or (None, x, y) (batch of matrices)
        :return:
        """
        z_m_train = self.encode(x_train)

        self.svm.fit(z_m_train)
        self.svm_fitted_ = True
        return self

    def fit_predict(self, x_train):
        z_m_train = self.encode(x_train)

        self.svm.fit(z_m_train)
        self.svm_fitted_ = True
        return self.binarize(self.svm.predict(z_m_train))

    def predict(self, x):
        z_m = self.encode(x)

        return self.binarize(self.svm.predict(z_m))

    def score_samples(self, x):
        """
        In sklearn convention, a lower score indicates an anomaly. In our convention this is
        the opposite.
        :return: anomaly score function on x
        """
        z_m = self.encode(x)
        score = self.svm.score_samples(z_m)
        return score

    def score(self, x, y_true):
        z_m = self.encode(x)
        y_pred = self.binarize(self.svm.predict(z_m))
        return roc_auc_score(y_true, y_pred)

    def decision_function(self, x):
        z_m = self.encode(x)
        return self.svm.decision_function(z_m)

    def plot_tSNE(self, x, y_true):
        z_m = self.encode(x)
        y_pred = self.svm.predict(z_m)
        x_embedded = TSNE(n_components=2).fit_transform(z_m)
        y_scores = self.score_samples(x)

        # choose a color palette with seaborn.
        num_classes = len(np.unique(y_true))
        palette = np.array(sns.color_palette("hls", num_classes))

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # create meshgrid
        resolution = 300  # 100x100 background pixels
        x2d_xmin, x2d_xmax = np.min(x_embedded[:, 0]), np.max(x_embedded[:, 0])
        x2d_ymin, x2d_ymax = np.min(x_embedded[:, 1]), np.max(x_embedded[:, 1])
        xx, yy = np.meshgrid(np.linspace(x2d_xmin, x2d_xmax, resolution), np.linspace(x2d_ymin, x2d_ymax, resolution))

        # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
        background_model = KNeighborsClassifier(n_neighbors=1).fit(x_embedded, y_pred)
        voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
        voronoiBackground = voronoiBackground.reshape((resolution, resolution))

        # plot
        ax.set_title("t-SNE projection of the points")
        plt.pcolormesh(xx, yy, voronoiBackground, cmap=plt.cm.Paired)
        #         plt.contour(xx, yy, voronoiBackground, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
        #                     levels=[-.5, 0, .5])
        ax.contourf(xx, yy, voronoiBackground, cmap=plt.cm.Paired)
        ax.scatter(x_embedded[:, 0], x_embedded[:, 1], c=palette[y_true.astype(np.int)])
