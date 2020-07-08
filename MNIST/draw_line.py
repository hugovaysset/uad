# Utility functions to add anomalies and plot them on images


import matplotlib.pyplot as plt
import numpy as np

flat_shape = 784
square_shape = 28


def set_anomaly(img, an_type="l", an_size=3, s=0):
    """
    Set a "draw line" anomaly on the given image
    an_type: "l": line, "c": circle, "s": square
    an_size: size of anomaly
    s: random seed
    """
    np.random.seed(s)
    square_shape = 28

    if img.shape == (square_shape ** 2,):  # flat format
        img = img.reshape((28, 28))

    if img.shape == (28, 28, 1):
        img = np.squeeze(img, axis=-1)

    modif_img = np.array(img)

    x, y = np.random.randint(0, square_shape, size=2)

    if an_type == "l":
        for i in range(an_size - 1):
            if x + i < square_shape and y + i + 1 < square_shape:
                modif_img[x + i, y + i] = .99
                modif_img[x + i, y + i + 1] = .5
                modif_img[x + i, y + i - 1] = .5

            if x - i >= 0 and y - i - 1 >= 0 and y - i + 1 < 28:
                modif_img[x - i, y - i] = .99
                modif_img[x - i, y - i + 1] = .5
                modif_img[x - i, y - i - 1] = .5

    if an_type == "c":
        for i in range(an_size):
            for j in range(an_size):
                if np.sqrt(i**2 + j**2) <= an_size:
                    if x + i < square_shape and y + j < square_shape:
                        modif_img[x + i, y + j] = 0.99
                    if x - i >= 0 and y + j < square_shape:
                        modif_img[x - i, y + j] = 0.99
                    if x + i < square_shape and y - j >= 0:
                        modif_img[x + i, y - j] = 0.99
                    if x - i >= 0 and y - j >= 0:
                        modif_img[x - i, y - j] = 0.99

    if an_type == "s":
        for i in range(an_size // 2):
            for j in range(an_size // 2):
                    if x + i < square_shape and y + j < square_shape:
                        modif_img[x + i, y + j] = 0.99
                    if x - i >= 0 and y + j < square_shape:
                        modif_img[x - i, y + j] = 0.99
                    if x + i < square_shape and y - j >= 0:
                        modif_img[x + i, y - j] = 0.99
                    if x - i >= 0 and y - j >= 0:
                        modif_img[x - i, y - j] = 0.99

    return modif_img


def predict_anomalies(model, ref, dims=(28, 28, 1)):
    """
    Make model predictions on reference and reference + anomalies
    tensor (bool): True if the model takes as inputs a rank-3 tensor (28, 28, 1)
    Assuming ref.shape == (28, 28, 1) initially
    """
    anom_types = ["l", "s", "c"]
    anom_sizes = [3, 4, 5]

    predictions = model.predict(ref)

    l = []
    for i, x in enumerate(ref):
        np.random.seed(i)
        t, size = np.random.choice(anom_types), np.random.choice(anom_sizes)
        anom = set_anomaly(x, an_type=t, an_size=size, s=i)
        l.append(anom)
    anomalies = np.array(l)

    print(anomalies.shape)

    anomalies_pred = np.array([])
    if dims == (784,):
        anomalies_pred = model.predict(np.reshape(anomalies, (anomalies.shape[0], 784,)))

    elif dims == (28, 28, 1):
        anomalies_pred = model.predict(np.expand_dims(anomalies, axis=-1))
        predictions = np.squeeze(predictions, axis=-1)
        anomalies_pred = np.squeeze(anomalies_pred, axis=-1)

    else:
        anomalies_pred = model.predict(anomalies)

    return predictions, anomalies, anomalies_pred


def contour_anomalies(img, maskimg, legend="anomaly"):
    """
    Draw contour line on the edges of the pixels identified by the autoencoder
    as the anomaly
    """
    mapimg = (maskimg.reshape((square_shape, square_shape)) == True)
    ver_seg = np.where(mapimg[:, 1:] != mapimg[:, :-1])
    hor_seg = np.where(mapimg[1:, :] != mapimg[:-1, :])

    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0] + 1))
        l.append((p[1] + 1, p[0] + 1))
        l.append((np.nan, np.nan))

    for p in zip(*ver_seg):
        l.append((p[1] + 1, p[0]))
        l.append((p[1] + 1, p[0] + 1))
        l.append((np.nan, np.nan))

    segments = np.array(l)  # array of size Nx2
    segments[:, 0] = square_shape * segments[:, 0] / mapimg.shape[1]
    segments[:, 1] = square_shape * segments[:, 1] / mapimg.shape[0]

    # and now there isn't anything else to do than plot it
    img.plot(segments[:, 0], segments[:, 1], color='red', linewidth=2, label=legend)
    img.legend()


def plot_anomalies(ref, pred, anomalies, anomalies_pred, show_idx=0, threshold=0.5, ref_dims=(28, 28, 1)):
    """
    Plot four images using matplotlib and contour the anomalies. Takes only 2D-arrays as inputs, if necessary
    remove the extra-axis using np.squeeze(ar, axis=-1)
    :param ref: ground truth image
    :param pred: prediction of the model on ref
    :param anomalies: image containing the anomaly
    :param anomalies_pred: prediction of the model on anomalies
    :param show_idx: index of the image to choose in the test set
    :param threshold: threshold for the contour of the predicted  anomalies
    :param ref_dims: if the model takes as input a rank-3 tensor, removes the width dim before plot
    :return:
    """
    fig, axis = plt.subplots(2, 3, figsize=(12, 8), sharex="all", sharey="all")

    if ref_dims == (28, 28, 1):
        ref = np.squeeze(ref, axis=-1)

    if ref_dims == (784,):
        ref = np.reshape(ref, (ref.shape[0], 28, 28))
        pred = np.reshape(pred, (ref.shape[0], 28, 28))
        anomalies = np.reshape(anomalies, (ref.shape[0], 28, 28))
        anomalies_pred = np.reshape(anomalies_pred, (ref.shape[0], 28, 28))

    true_anomaly = np.abs(ref[show_idx] - anomalies[show_idx]) > 0
    predicted_anomaly = np.abs(anomalies_pred[show_idx] - anomalies[show_idx]) > threshold

    axis[0][0].imshow(ref[show_idx])
    axis[0][0].set_title("Original")

    axis[0][1].imshow(pred[show_idx])
    axis[0][1].set_title("Prediction on original")

    axis[0][2].imshow(np.abs(pred[show_idx] - ref[show_idx]))
    axis[0][2].set_title("L1 residual map |original - prediction|")

    axis[1][0].imshow(anomalies[show_idx])
    axis[1][0].set_title("Image wit anomaly")

    axis[1][1].imshow(anomalies_pred[show_idx])
    axis[1][1].set_title("Prediction on anomaly")

    axis[1][2].imshow(np.abs(anomalies_pred[show_idx] - anomalies[show_idx]))
    axis[1][2].set_title("L1 residual map |anomaly - prediction_anomaly|")

    contour_anomalies(axis[1][0], true_anomaly, legend="GT anomaly")
    contour_anomalies(axis[1][1], predicted_anomaly, legend="Predicted anomalies")


def get_rm(im1, im2, rm_type="l2"):
    """
    rm_type: "l2" or "l1"
    Returns: Residual map and mean loss between im1, im2
    """
    rm = np.array([])
    if im1.shape == (28, 28, 1):
        im1 = np.squeeze(im1, axis=-1)
    if im2.shape == (28, 28, 1):
        im2 = np.squeeze(im2, axis=-1)
    if rm_type == "l2":
        rm = (im1 - im2) ** 2
    elif rm_type == "l1":
        rm = np.abs(im1 - im2)
    return rm, np.mean(rm)
    


def plot_predictions(model, inputs, n=5, dims=(28, 28, 1)):
    plt.figure(figsize=(25, 8))
    plt.viridis()

    predictions = model.predict(inputs[:n].reshape((n, *dims)))

    if dims == (28, 28, 1):
        inputs = np.squeeze(inputs, axis=-1)

    for i in range(n):
        # plot original image
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(inputs[i].reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original Images')

        # plot reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(predictions[i].reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed Images')

        # plot residual maps
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(get_rm(inputs[i], predictions[i], rm_type="l1")[0].reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Residual Maps')
    plt.show()

