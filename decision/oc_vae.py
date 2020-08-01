import tensorflow as tf


def anomaly_score(model, data, decision_func="distance", batch=True):
    """
    Compute the anomaly score from preidctions of model. Those predictions are considered
    to vectors
    # TODO: implement for matrix or 3-tensors predictions
    :param model: Model that has a CENTER attribute
    :param data: input data in tf format (3-tensor)
    :param decision_func: can be either "distance" to predict anomalies based on their distance to the model's center
    (in an SVDD manner) or "reconstruction" to predict anomalies based on the reconstruction error between the input
    image and the reconstruction (using MSE, in a VAE manner).
    :param batch: True if the given data is a batch
    :return: a vector of distance or a distance depending on the input data
    """
    if decision_func == "distance":
        predictions, z_log_var, _ = model.encoder.predict(data)
        if batch:
            if len(predictions.shape) == 4:  # batch of vectors along the first axis
                return tf.math.sqrt(tf.reduce_sum((model.CENTER - predictions) ** 2, axis=(-3, -2, -1)))
            elif len(predictions.shape) == 3:  # batch of matrices along the first axis
                return tf.math.sqrt(tf.reduce_sum((model.CENTER - predictions) ** 2, axis=(-2, -1)))
            elif len(predictions.shape) == 2:  # batch of vectors along the first axis
                return tf.math.sqrt(tf.reduce_sum((model.CENTER - predictions) ** 2, axis=-1))
        else:
            return tf.math.sqrt(tf.reduce_sum((model.CENTER - predictions) ** 2))

    if decision_func == "reconstruction":
        predictions = model.predict(data)
        return tf.math.sqrt(tf.reduce_sum((data - predictions) ** 2, axis=(-3, -2, -1)))

