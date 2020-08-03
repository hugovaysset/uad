import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from uad.models.self_attention import SelfAttention
from uad.models.variational_autoencoder import Sampling


def conv2d_block(input_tensor, n_filters, kernel_size=(3, 1), batchnorm=True,
                 activation="relu"):
    """Function to add 2 convolutional layers with the parameters passed to it
    activation1: name of the activation function to apply. If none, pass "" (empty string)
    activation2: name of the activation function to apply. If none, pass "" (empty string)
    """
    # first layer
    x = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                      kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation == "relu" or activation == "sigmoid" or activation == "linear":
        x = layers.Activation(activation)(x)
    elif activation == "leaky_relu" or activation == "l_relu" or activation == "lrelu":
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    else:
        raise NotImplementedError("activation function should be given by a valid string of leaky_relu")

    # second layer
    x = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                      kernel_initializer='he_normal', padding='same')(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    if activation == "relu" or activation == "sigmoid" or activation == "linear":
        x = layers.Activation(activation)(x)
    elif activation == "leaky_relu":
        x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    else:
        raise NotImplementedError("activation function should be given by a valid string of leaky_relu")

    return x


def get_unet_vae(n_filters=64, n_contractions=3, input_dims=(28, 28, 1), k_size=(3, 3), batchnorm=False, dropout=0,
                 spatial_dropout=0.2, padding=None, activation_function="relu", latent_depth=None, self_attention=False,
                 final_activation=""):
    """
    U-Net architecture is composed of a contraction paths ((1 convolution layers, 1 activation layer)**2, 1 max pooling)**n
     and one expansive path: (1 convolution transpose, (1 convolution layers, 1 activation layer)**2)**n terminated by
     a single convolution
    :param n_filters:
    :param n_contractions:
    :param inputs_dims:
    :param k_size:
    :param batchnorm:
    :param dropout:
    :param spatial_dropout:
    :param padding: if padding is needed in the beginning, pass into a numpy array the padding to apply see Keras doc
    :param final_activation: if not "", apply an activation function as the final layer. Specify in a string the name
    of the final activation function
    :return:
    """

    if latent_depth is None:
        latent_depth = n_filters * int(2 ** n_contractions)
    if padding is None:
        latent_dims = (int(input_dims[0] / (2 ** n_contractions)),
                       int(input_dims[1] / (2 ** n_contractions)), latent_depth)
    else:
        latent_dims = (int((input_dims[0] + padding[1][0] + padding[1][1]) / (2 ** n_contractions)),
                       int((input_dims[1] + padding[2][0] + padding[2][1]) / (2 ** n_contractions)), latent_depth)

    encoder_inputs = layers.Input(shape=input_dims, name="encoder_inputs")

    if padding is not None:
        print(padding)
        paddings = tf.constant(padding)  # shape d x 2 where d is the rank of the tensor and
        # 2 represents "before" and "after"
        x = tf.pad(encoder_inputs, paddings, name="pad")

    # contracting path
    for i in range(n_contractions):
        if i == 0 and padding is None:
            x = conv2d_block(encoder_inputs, n_filters * 2 ** i, kernel_size=k_size,
                             batchnorm=batchnorm, activation=activation_function)
        else:
            x = conv2d_block(x, n_filters * 2 ** i, kernel_size=k_size, batchnorm=batchnorm,
                             activation=activation_function)
        x = layers.MaxPooling2D((2, 2))(x)
        if dropout != 0:
            x = layers.Dropout(dropout)(x)
        if spatial_dropout != 0:
            x = layers.SpatialDropout2D(rate=dropout)(x)

    if self_attention:
        s = SelfAttention(latent_depth // 2, (latent_dims[0], latent_dims[1]), positional_encoding=True)(x)[0]
        x = layers.Concatenate(axis=-1)([s, x])

    z_mean = layers.Conv2D(latent_depth, 1, strides=1, name="z_mean")(x)
    z_log_var = layers.Conv2D(latent_depth, 1, strides=1, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Define decoder model.
    latent_inputs = layers.Input(shape=latent_dims, name="z_sampling")

    for i in range(n_contractions - 1, 0, -1):
        if i == n_contractions - 1:
            x = layers.Conv2DTranspose(n_filters * 2 ** i, k_size, strides=(2, 2),
                                       padding='same')(latent_inputs)
        else:
            x = layers.Conv2DTranspose(n_filters * 2 ** i, k_size, strides=(2, 2),
                                       padding='same')(x)
        x = conv2d_block(x, n_filters * 2 ** i, kernel_size=k_size, batchnorm=batchnorm,
                         activation=activation_function)
        if dropout != 0:
            x = layers.Dropout(dropout)(x)
        if spatial_dropout != 0:
            x = layers.SpatialDropout2D(rate=dropout)(x)

    x = layers.Conv2DTranspose(n_filters, kernel_size=k_size, strides=(2, 2),
                               padding='same')(x)
    if dropout != 0:
        x = layers.Dropout(dropout)(x)
    if spatial_dropout != 0:
        x = layers.SpatialDropout2D(rate=dropout)(x)

    x = conv2d_block(x, input_dims[-1], kernel_size=k_size, batchnorm=batchnorm, activation=activation_function)

    if padding is not None:
        x = tf.image.resize_with_crop_or_pad(x, input_dims[0], input_dims[1])

    if final_activation != "":
        x = layers.Activation(final_activation)(x)

    decoder = Model(inputs=latent_inputs, outputs=x, name="decoder")

    return encoder, decoder


def get_ruff_svdd(input_dims=(32, 32, 3), n_filters=(32, 64, 128), dense_sizes=(64, 2), k_size=(5, 5), LAMBDA=1e-6,
                  spatial_dropout=0.2, dropout=0, batchnorm=False):
    """
    LeNet-type architecture descirbed by Ruff et al. in their publication
    :param input_dims:
    :param n_filters:
    :param dense_sizes:
    :param k_size:
    :param LAMBDA:
    :param spatial_dropout:
    :param dropout:
    :param batchnorm:
    :return:
    """
    inputs = tf.keras.Input(shape=input_dims)

    # convolutions
    n_blocks = len(n_filters)
    for i in range(n_blocks):
        if i == 0:
            x = layers.Conv2D(filters=n_filters[0], kernel_size=k_size, strides=(1, 1), kernel_regularizer=l2(LAMBDA),
                              padding="same", name=f"conv_1", use_bias=False)(inputs)
        else:
            x = layers.Conv2D(filters=n_filters[0], kernel_size=5, strides=(1, 1), kernel_regularizer=l2(LAMBDA),
                              padding="same", use_bias=False)(x)
        if spatial_dropout != 0:
            x = layers.SpatialDropout2D(spatial_dropout)(x)
        if dropout != 0:
            x = layers.Dropout(dropout)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    # denses
    n_denses = len(dense_sizes)
    for i in range(n_denses):
        x = layers.Dense(dense_sizes[i], kernel_regularizer=l2(LAMBDA), use_bias=False)(x)

    return tf.keras.Model(inputs, x)


def get_ruff_vae(input_dims=(32, 32, 3), n_filters=(32, 64, 128), k_size=(5, 5), LAMBDA=1e-6,
                 spatial_dropout=0.2, dropout=0, batchnorm=False):
    """
    LeNet-type architecture descirbed by Ruff et al. in their publication
    :param input_dims:
    :param n_filters:
    :param dense_sizes:
    :param k_size:
    :param LAMBDA:
    :param spatial_dropout:
    :param dropout:
    :param batchnorm:
    :return:
    """
    inputs = tf.keras.Input(shape=input_dims)

    # contraction
    n_contractions = len(n_filters)
    for i in range(n_contractions):
        if i == 0:
            x = layers.Conv2D(filters=n_filters[i], kernel_size=k_size, strides=(1, 1), kernel_regularizer=l2(LAMBDA),
                              padding="same", name=f"conv_1")(inputs)
        else:
            x = layers.Conv2D(filters=n_filters[i], kernel_size=k_size, strides=(1, 1), kernel_regularizer=l2(LAMBDA),
                              padding="same")(x)
        if spatial_dropout != 0:
            x = layers.SpatialDropout2D(spatial_dropout)(x)
        if dropout != 0:
            x = layers.Dropout(dropout)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling2D((2, 2))(x)

    latent_depth = n_filters[0] * int(2 ** n_contractions)
    latent_dims = (int(input_dims[0] / (2 ** n_contractions)), int(input_dims[1] / (2 ** n_contractions)), latent_depth)

    z_mean = layers.Conv2D(latent_depth, 1, strides=1, name="z_mean")(x)
    z_log_var = layers.Conv2D(latent_depth, 1, strides=1, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))

    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # expansion
    latent_inputs = tf.keras.Input(shape=latent_dims)

    for i in range(n_contractions - 1, -1, -1):
        if i == n_contractions - 1:
            x = layers.Conv2DTranspose(n_filters[i], kernel_size=k_size, strides=(2, 2))(latent_inputs)
        else:
            x = layers.Conv2DTranspose(n_filters[i], kernel_size=k_size, strides=(2, 2))(x)
        x = layers.Conv2D(filters=n_filters[i], kernel_size=k_size, strides=(1, 1), kernel_regularizer=l2(LAMBDA),
                          padding="same")(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        if spatial_dropout != 0:
            x = layers.SpatialDropout2D(spatial_dropout)(x)
        if dropout != 0:
            x = layers.Dropout(dropout)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=input_dims[-1], kernel_size=k_size, strides=(1, 1), kernel_regularizer=l2(LAMBDA),
                      padding="same")(x)

    decoder = Model(latent_inputs, x, name="encoder")

    return encoder, decoder
