import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import roc_curve, auc
from uad.decision.reconstruction import decision_function


def is_binary(labels):
    l = []
    for elt in labels:
        if not elt in l:
            l.append(elt)
    return len(l) == 2


def binarize_set(labels, interest=0):
    l = []
    for elt in labels:
        if elt == interest:
            l.append(0)
        else:
            l.append(1)
    return np.array(l)


def compute_ROC(gt_data, predictions, gt_labels, criterion="threshold", pix_threshold=0.5, interest_class=0,
                normalized=False):
    """
    Plot the ROC curve of a model based on its predictions on a give dataset.
    :param gt_data: Ground truth dataset
    :param gt_labels: Ground truth labels on the dataset (BINARIZED!)
    :param predictions: model predictions on the dataset
    :param interest_class: In case binarization is needed, interest_class is the normal (i.e. 0)
    """
    labels_pred = decision_function(gt_data, predictions, criterion=criterion, pix_threshold=pix_threshold,
                                    normalized=normalized).ravel()
    if not is_binary(gt_labels):
        gt_labels = binarize_set(gt_labels, interest=interest_class)
    fpr, tpr, thresholds = roc_curve(gt_labels, labels_pred)
    return fpr, tpr, thresholds


def compute_AUC(fpr, tpr):
    return auc(fpr, tpr)


def compute_AUC_on_all(model_class, x_train, x_test, y_test, n_classes=10, epochs=8, **params):
    """
    Successively train a particular model on each class (considering this class as normal and all the others as abnormal)
    and evaluate it by computing the AUC score in each case.
    :param model_class: class which implements the model
    :param x_train: training data sorted by digits ([all_0_examples, ..., all_n_examples])
    :param x_test: test set (containing examples of all digits)
    :param y_test: test labels (don't have to binary)
    :param n_classes: number of classes/digits
    :param epochs: number of epochs of training
    """
    auc_scores = []
    for k in range(n_classes):
        print(f"Digit {k}, # Training examples: {x_train[k].shape[0]}")
        model = model_class(**params)
        model.compile(optimizer=tf.keras.optimizers.Adam())
        model.fit(x_train[k], x_train[k], epochs=epochs, batch_size=128)
        predictions = model.predict(x_test)
        fpr, tpr, _ = compute_ROC(x_test, predictions, y_test, criterion="l2", interest_class=k)
        auc = compute_AUC(fpr, tpr)
        auc_scores.append(auc)

    return np.array(auc_scores)


def score_samples_iterator(model, dataset_iterator):
    """
    Compute scores (mse) between images and predictions.
    :param model: generic model, has to implement a predict() method
    :param dataset_iterator: iterator of MultiChannelIterator type
    Return: scores in the batch format
    """
    scores = []
    for i in range(len(dataset_iterator)):  # itere a l'infini???
        (ims, labs), _ = dataset_iterator[i]
        if i % 50 == 0:
            print(f"making predictions on batch {i}...")
        predictions = model.predict(ims)
        y_scores = np.sum((predictions - ims) ** 2, axis=(1, 2, 3))
        scores.append(y_scores)

    return np.array(scores)


def compute_ROC_iterator(model, dataset_iterator, interest_digit=7):
    """
    :param dataset_iterator: expected to be in the format given by
    MultiChannelIterator with (bx, by) == dataset_iterator[0] and then
    (images_x, labels_x) == bx
    """
    labels = []
    for i in range(len(dataset_iterator)):
        (ims, y_true), _ = dataset_iterator[i]
        labels.append(y_true.squeeze(-1))
    y_trues = np.array(labels).flatten()

    y_scores = score_samples_iterator(model, dataset_iterator).flatten()

    if not is_binary(y_trues):
        y_true_bin = binarize_set(y_trues, interest=interest_digit)
    else:
        y_true_bin = y_trues

    fpr, tpr, thresholds = roc_curve(y_true_bin, y_scores)

    return fpr, tpr, thresholds


def plot_ROC(fpr, tpr, labels=["ROC curve"]):
    """
    Plot the ROC curves from false positives rate and true positives rate
    :param fpr: array: false positive rate
    :param tpr: array: true positive rate
    :param labels: str or list: labels to give to each curve
    :return:
    """
    if type(fpr) == list:
        fpr = np.array(fpr)
    if type(tpr) == list:
        tpr = np.array(tpr)
    if type(labels) == list:
        labels = np.array(labels)

    colours = ['r', 'g', 'b', 'm', 'y', 'k']
    symbols = ['*', '.', '-', '--', '^', 'v']
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), sharex="all", sharey="all")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    fig.suptitle("ROC curve")
    if len(fpr.shape) == 1:  # single curve
        ax.plot(fpr, tpr, '.', c="orange")
        ax.plot(fpr, tpr, c="orange", label=f"{labels[0]} (AUC = {round(compute_AUC(fpr, tpr), 3)})")
    else:  # several curves to plot
        for i, (f, t, lab) in enumerate(zip(fpr, tpr, labels)):
            print(f.shape, t.shape)
            ax.plot(f, t, ".", f"{colours[i % len(colours)]}.")
            ax.plot(f, t, f"{colours[i % len(colours)]}.", label=f"{lab} (AUC = {round(compute_AUC(f, t), 3)})")
    ax.legend()
    ax.set_yscale("linear")
    ax.set_ylim(bottom=-0.02, top=1.03)
    return fig, ax


def plot_history(history, metric_names=["reconstruction_loss", "validation_loss", "validation_accuracy"]):
    """
    Plot the values of given loss/metric during train step. Group the similar metrics by row in order to
    have separate plots (better if the scales of each metrics are different)
    """
    colors = ["b", "g", "orange", "k", "c", "m", "y"]

    fig, axes = plt.subplots(1, metric_names.shape[0], figsize=(20, 8), sharex="all")
    if metric_names.shape[0] == 1:  # 1 row, several columns
        ax = axes  # do not support iteration if there is one element
        for i, metric in enumerate(metric_names[0]):  # iterate on the first row
            ax.plot(history.history[metric], c=colors[i % len(colors)], label=metric_names[0][i])
        ax.legend()
        ax.set_title("Loss/metric during train step")
        ax.set_ylabel("loss/metric")
        ax.set_xlabel("epoch")
    elif metric_names.shape[0] > 1 and len(metric_names.shape) == 1:  # several metrics in one vector
        ax = axes
        for i, metric in enumerate(metric_names):
            print(colors[i % len(colors)])
            print(history.history[metric])
            ax.plot(history.history[metric], c=colors[i % len(colors)], label=metric_names[0][i])
        ax.legend()
        ax.set_title("Loss/metric during train step")
        ax.set_ylabel("loss/metric")
        ax.set_xlabel("epoch")
    else:  # several rows, columns
        for row, ax in zip(metric_names, axes):
            for i, metric in enumerate(row):
                print(colors[i % len(colors)])
                print(history.history[metric])
                ax.plot(history.history[metric], c=colors[i % len(colors)], label=row[i])
            ax.legend()
            ax.set_title("Loss/metric during train step")
            ax.set_ylabel("loss/metric")
            ax.set_xlabel("epoch")

    return fig, axes


def compute_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = tf.cast(true_positives, dtype=tf.float32) / (tf.cast(possible_positives, dtype=tf.float32) + K.epsilon())
    return recall


def compute_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = tf.cast(true_positives, dtype=tf.float32) / (
                tf.cast(predicted_positives, dtype=tf.float32) + K.epsilon())
    return precision
