import h5py
import numpy as np
# import tensorflow as tf
import tensorflow.image as image
from tensorflow.keras.preprocessing import ImageDataGenerator


def load_MNIST(scale=True, expand_dims=True, validation_set=True, data_augmentation=True, binarize=False, interest_digit=0):
    train, test = h5py.File("dataset/train.hdf5", 'r'), h5py.File("dataset/test.hdf5", 'r')
    (x_train, y_train), (x_test, y_test) = (train["image"], train["label"]), (test["image"], test["label"])
    x_val, y_val = np.array([]), np.array([])

    if expand_dims:
        x_train = np.expand_dims(x_train, -1).astype("float32")
        x_test = np.expand_dims(x_test, -1).astype("float32")

    if scale:
        x_train, x_test = x_train / 255.0, x_test / 255.0

    if binarize:
        digits_train = np.array([x_train[np.where(y_train[:-1000] == i)] for i in range(10)])
        x_train, y_train, y_test = digits_train[interest_digit], binarize(y_train), y_test

    if validation_set:
        x_val, y_val = x_train[-1000:], y_train[-1000:]
        x_train, y_train = x_train[:-1000], y_train[:-1000]

    if data_augmentation:
        x_train = perform_augmentation(x_train)

    return x_train, y_train, x_test, y_test, x_val, y_val


def data_augmentation(dataset, batch_size=128, rescale=False, rotation_range=15, width_shift=0.15, height_shift=0.15,
                      horizontal_flip=True):
    aug = ImageDataGenerator(rescale=rescale, rotation_range=rotation_range, width_shift=width_shift,
                              height_shift=height_shift, horizontal_flip=horizontal_flip).fit(dataset, augment=True)
    image_gen = aug.flow(dataset, batch_size=batch_size)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255

    x_val, y_val = x_train[-1000:], binarize_set(y_train[-1000:], interest=0)

    interest_digit = 2

    # train set sorted by digits: digits_train[i] = x_train elements where y_train == i
    digits_train = np.array([x_train[np.where(y_train[:-1000] == i)] for i in range(10)])

    # training set contains only zeros (for training on zeros)
    x_train_interest = digits_train[interest_digit][:-1000]
    y_test_bin = binarize_set(y_test, interest=interest_digit)

    params = dict(rescale=False, rotation_range=15, width_shift_range=0.15,
                  height_shift_range=0.15, horizontal_flip=True)
    aug = ImageDataGenerator(**params)
    aug.fit(x_train_interest)
    aug.flow(x_train_interest, save_to_dir=["dataset"], save_prefix="aug", save_format="jpg")




