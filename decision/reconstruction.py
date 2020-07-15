import numpy as np


def is_anormal(reference, predictions, criterion="threshold", pix_threshold=0.5, im_threshold=10):
    """
    Takes images as inputs! Predicts if each image of predictions is an anomaly according to the criterion
    and the reference images.
    reference: ground truth IMAGES
    predictions: predicted IMAGES
    Return: An array of the size of reference/predictions containing 0 for normal class and 1 for anomalies. Typical
    values for im_threshold have to be tuned on the evaluation set ("threshold" -> 10, "L1" -> ?, "L2" -> ?
    """
    if reference.shape == (reference.shape[0], 28, 28, 1):
        reference1 = np.squeeze(reference, axis=-1)
    else:
        reference1 = reference
    if predictions.shape == (predictions.shape[0], 28, 28, 1):
        predictions1 = np.squeeze(predictions, axis=-1)
    else:
        predictions1 = predictions

    # detection via reconstruction
    if criterion == "threshold":
        return binarize_booleans(threshold_criterion(reference1, predictions1, pix_threshold) > im_threshold)

    if criterion == "l1":
        return binarize_booleans(L1_criterion(reference1, predictions1) > im_threshold)

    if criterion == "l2":
        return binarize_booleans(L2_criterion(reference1, predictions1) > im_threshold)


def decision_function(reference, predictions, criterion="threshold", pix_threshold=0.5, normalized=False):
    """
    Same as is_normal, but returns the value of the decision function instead. Notably used in the computation of the
    ROC curve
    :param reference:
    :param predictions:
    :param criterion:
    :param pix_threshold:
    :return: an array of the anormality scores of each prediction/reference given
    """
    if reference.shape == (reference.shape[0], 28, 28, 1):
        reference1 = np.squeeze(reference, axis=-1)
    else:
        reference1 = reference
    if predictions.shape == (predictions.shape[0], 28, 28, 1):
        predictions1 = np.squeeze(predictions, axis=-1)
    else:
        predictions1 = predictions

    # detection via reconstruction
    if criterion == "threshold":
        return threshold_criterion(reference1, predictions1, pix_threshold, normalized)

    if criterion == "l1":
        return L1_criterion(reference1, predictions1)

    if criterion == "l2":
        return L2_criterion(reference1, predictions1)


# validation set contains a bit of all digits
def binarize_set(ar, interest=0):
    """
    Takes a multiclass array (e.g. MNIST datasets) to a binary array with 0 representing
    the normal class (interest) and 1 representing all the others (anomalies)
    :param ar:
    :param interest:
    :return:
    """
    l = []
    for elt in ar:
        if elt == interest:  # class of interest = non-anormal
            l.append(0)
        else:  # anomaly
            l.append(1)
    return np.array(l)


def binarize_booleans(booleans):
    """
    Transform a given array of booleans to an array containing 0 and 1.
    :param booleans:
    :return:
    """
    l = []
    for b in booleans:
        if b:
            l.append(1)
        else:
            l.append(0)
    return np.array(l)


def threshold_criterion(reference, predictions, pix_threshold=0.5, normalized=False):
    """
    Return: the number
    """
    l = []
    for pred, ref in zip(predictions, reference):
        residual_map = np.abs(ref - pred)
        if normalized:
            l.append(np.sum(residual_map[residual_map > pix_threshold]) / np.sum(ref[ref > pix_threshold]))
        else:
            l.append(np.sum(residual_map[residual_map > pix_threshold]))
    return np.array(l)


def L2_criterion(reference, predictions):
  """
  Returns the L2 distance between two images. Reference and predictions should
  be either images (matrix) of batch of images (3-tensor)
  """
  if len(reference.shape) == 2: # individual example
    return np.sum((reference - predictions) ** 2)
  elif len(reference.shape) == 3: # batch
    return np.sum((reference - predictions) ** 2, axis=(1, 2))



def L1_criterion(reference, predictions):
  """
  Returns the L1 distance between two images. Reference and predictions should
  be either images (matrix) of batch of images (3-tensor)
  """
  if len(reference.shape) == 2: # individual example
    return np.sum(np.abs(reference - predictions))
  elif len(reference.shape) == 3: # batch
    return np.sum(np.abs(reference - predictions), axis=(1, 2))
