import tensorflow as tf


def anomaly_score_from_predictions(model, predictions):
    """
    Compute the anomaly score from preidctions of model. Those predictions are considered
    to vectors
    # TODO: implement for matrix or 3-tensors predictions
    :param model: Model that has a CENTER attribute
    :param predictions: Model predictions
    :return:
    """
    if len(predictions.shape) > 1:  # batch of vectors along the first axis
        return tf.norm(model.CENTER - predictions, axis=-1) ** 2
    else:  # single vector, tf.norm parameter axis=None
        return tf.norm(model.CENTER - predictions) ** 2


def anomaly_score_from_images(model, images, n_images_axis=3):
    """
    Compute the anomaly score output by a model from a single image or a batch of images
    :param model: Deep SVDD model that possedes a self.CENTRE attribute
    :param images: either single image or batch of images
    :param n_images_axis: (int) Number of axis of the images/batch of images. If a single image
    is a 3-tensor (tf default format: (x, y, channels)) input 3, if images are matrices (x, y)
    input 2.
    :return:
    """
    if n_images_axis == 3 and len(images.shape) == 4:  # batch of images (x, y, channels) along the first axis
        return tf.norm(tf.norm(model.CENTER - images, axis=(-2, -1)), axis=-1) ** 2
    elif n_images_axis == 3 and len(images.shape) == 2:  # single image
        return tf.norm(model.CENTER - images, axis=(-2, -1)) ** 2
    elif n_images_axis == 2 and len(images.shape) == 3:
        return tf.norm(model.CENTER - images, axis=(-2, -1)) ** 2
    elif n_images_axis == 2 and len(images.shape) == 2: # single matrix
        return tf.norm(model.CENTER - images, axis=(-2, -1)) ** 2
    else:
        raise NotImplementedError

def is_anormal(model, predictions, threshold):
    pass