import matplotlib.pyplot as plt
import numpy as np


def plot_prediction(ref, pred):
    fig, ax = plt.subplots(1, 3, figsize=(10, 10), sharex="all", sharey="all")

    ax[0].imshow(ref)
    ax[0].set_title("GT")

    ax[1].imshow(pred)
    ax[1].set_title("Prediction")

    ax[2].imshow(np.abs(pred - ref))
    ax[2].set_title("L1 Residual Map")

    return fig, ax


def plot_per_digit_proportion(reference, predictions, ref_labels, criterion="threshold", pix_threshold=0.5, im_threshold=10):
    """
    Plot the proportion of each digit predicted as anormal according to the models predictios and the chosen criterion
    """
    counts = [0 for i in range(10)]

    for i in range(10):
        mask = np.where(ref_labels == i)
        ref_sel, pred_sel = reference[mask], predictions[mask]
        size = ref_sel.shape[0]
        counts[i] = sum(1 for elt in is_anormal(ref_sel, pred_sel, criterion, pix_threshold, im_threshold) if elt) / size

    plt.plot(counts, ".")
    plt.suptitle("Proportion of predicted anomalies per digit")