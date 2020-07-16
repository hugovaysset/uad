#########################################################################
# Not supported anymore : see uad.diagnostic.metrics for updated version#
#########################################################################


import matplotlib.pyplot as plt
import numpy as np
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


def plot_ROC(fpr, tpr, labels=["ROC curve"]):
    """
    Plot the ROC curves from false positives rate and true positives rate
    :param fpr: array: false positive rate
    :param tpr: array: true positive rate
    :param labels: str or list: labels to give to each curve
    :return:
    """
    colours = ['r', 'g', 'b', 'm', 'y', 'k']
    symbols = ['*', '.', '-', '--', '^', 'v']
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), sharex="all", sharey="all")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    fig.suptitle("ROC curve")
    if fpr.shape[0] == 1:
        ax.plot(fpr, tpr, '.', c="orange")
        ax.plot(fpr, tpr, c="orange", label=f"{labels[0]} (AUC = {round(compute_AUC(fpr, tpr), 3)})")
    else:
        for i, (f, t, lab) in enumerate(zip(fpr, tpr, labels)):
            ax.plot(f, t, ".", f"{colours[i % len(colours)]}.")
            ax.plot(f, t, f"{colours[i % len(colours)]}.", label=f"{lab} (AUC = {round(compute_AUC(f, t), 3)})")
    ax.legend()
    ax.set_yscale("linear")
    ax.set_ylim(bottom=-0.02, top=1.03)
    return fig, ax


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
