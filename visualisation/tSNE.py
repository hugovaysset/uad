import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import time

from sklearn.manifold import TSNE

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123


def compute_tSNE(dataset, desired_axis=-1):
    """
    Performs tSNE projection on the given dataset, along the desired axis. If a
    third-rank is given, takes the mean of the other axis, to get a final vector
    along the desired axis (since t-SNE only takes vectors as inputs and we don't
    want to mix the different axis)
    :param dataset: np.array
    :param desired_axis: axis on which to project the dataset (e.g. to project
    on channels : axis=-1 in channels_last configuration). None or 0 if you want to
    merge all axis in one.
    :return: an array containing the same number of items than in the given
    dataset but with the output dimension of the t-SNE transformation
    """
    if len(dataset.shape) > 2:  # third-rank tensor or matrix
        if desired_axis is None or desired_axis == 0:
            if len(dataset.shape) == 4:
                batch, x, y, z = dataset.shape
                dataset = np.reshape(dataset, (batch, x * y * z))
            elif len(dataset.shape) == 3:
                batch, x, y = dataset.shape
                dataset = np.reshape(dataset, (batch, x * y))
            else:
                raise NotImplementedError("Input dataset should either be 3 or 2-rank tensor")
        else:  # merge only non desired axis by taking the mean along each
            axes = range(len(dataset.shape))
            axes = np.delete(axes, desired_axis)
            axes = np.delete(axes, 0)  # keep first axis (individual elements)
            while axes != []:
                dataset = np.mean(dataset, axis=axes[0])  # perform mean along the non-desired axis
                axes = [a - 1 for a in axes]
                del axes[0]
    print(f"t-SNE inputs shape: {dataset.shape}")
    return TSNE().fit_transform(dataset)


def plot_tSNE(dataset, colors, axis=-1, plot_center={"bary": False, "center": False}, plt_ax=None):
    """
    Plot the t-SNE projection of a given dataset
    :param dataset: t-SNE outputs
    :param colors: original labels which serve as colors
    :param plot_center:
    :param axis: axis along which to perform t-SNE
    :param plt_ax: matplotlib axis object on which to plot the result
    :return: matplotlib figure, axis, scatter and texts
    """
    x = compute_tSNE(dataset, desired_axis=axis)

    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    #     palette = np.array(colors)

    # create a scatter plot.
    if plt_ax is None:
        f = plt.figure(figsize=(8, 8))
        plt_ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt_ax.axis('on')
        plt_ax.axis('tight')
    else:
        f = None
        sc = plt_ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt_ax.axis('on')
        plt_ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = plt_ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    # plot the center
    if np.any(plot_center["bary"]):
        plt_ax.plot(plot_center["bary"][0], plot_center["bary"][1], "x", c="r", markersize=18)
        plt_ax.text(plot_center["bary"][0], plot_center["bary"][1], "bary", fontsize=20)
    if np.any(plot_center["center"]):
        plt_ax.plot(plot_center["center"][0], plot_center["center"][1], "x", c="b", markersize=18)
        plt_ax.text(plot_center["center"][0], plot_center["center"][1], "center", fontsize=20)

    return f, plt_ax, sc, txts


def get_barycentre(predictions, axis=-1, output_shape=(1, 4 * 4 * 8)):
    """
    Returns the barycentre of a given dataset
    :param predictions: dataset on which to compute the barycentre
    :param axis: axis along which to compute the barycentre (e.g. -1 for channels...)
    :param output_shape: desired output shape for the given tensor
    :return: well-shaped barycentre
    """
    if type(predictions) == np.array or type(predictions) == list:
        return np.mean(predictions, axis=axis).reshape(output_shape)
    elif type(predictions) == tf.Tensor:
        return tf.reshape(tf.mean(predictions, axis=axis), output_shape)
    else:
        raise None


def append_barycentre(predictions, axis=-1, bary_shape=(1, 4 * 4 * 8)):
    bary = get_barycentre(predictions, axis=axis, output_shape=bary_shape)
    return np.concatenate((predictions, bary), axis=0)
