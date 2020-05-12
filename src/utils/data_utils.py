import gzip
import numpy as np
import skimage.morphology
import skimage.filters


def extract_data(filename, image_shape, image_number):
    """
    Extract MNIST data as a Numpy array from the byte file.
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data

def binarize_MNIST(im):
    """
    Preprocess the MNIST images to yield a binary coherent mask of the number.
    """
    # grayscale morphological cleaning
    SE = skimage.morphology.square(2)
    im_out = skimage.morphology.closing(im, selem=SE)
    im_out = skimage.morphology.opening(im_out, selem=SE)
    # binarize with otsu
    t = skimage.filters.threshold_otsu(im_out)
    im_out = skimage.filters.apply_hysteresis_threshold(im_out, 0.9*t, t)

    return im_out

def extract_labels(filename, image_number):
    """
    Extract MNIST lables as a Numpy array from the byte file.
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def get_MNIST(data_path, digits=[0,1,2,3,4,5,6,7,8,9], image_shape=(28,28),
              train_size=60000, test_size=10000, binary=False):
    """
    Get the MNIST dataset for the required digits from the provided data folder
    path. Binarize the image if required.
    """
    assert all([d in list(range(10)) for d in digits]), 'Digit must be a list of valid digit (0,1,2,3,4,5,6,7,8,9).'

    train_images = extract_data(data_path+'train-images-idx3-ubyte.gz', image_shape, train_size)
    test_images = extract_data(data_path+'t10k-images-idx3-ubyte.gz', image_shape, test_size)
    train_labels = extract_labels(data_path+'train-labels-idx1-ubyte.gz', train_size)
    test_labels = extract_labels(data_path+'t10k-labels-idx1-ubyte.gz', test_size)

    digit_mask_train = np.isin(train_labels, digits)
    train_images = train_images[digit_mask_train]
    train_labels = train_labels[digit_mask_train]

    digit_mask_test = np.isin(test_labels, digits)
    test_images = test_images[digit_mask_test]
    test_labels = test_labels[digit_mask_test]

    if binary:
        train_images = np.stack([binarize_MNIST(im) for im in train_images], axis=0)
        test_images = np.stack([binarize_MNIST(im) for im in test_images], axis=0)

    return train_images, train_labels, test_images, test_labels
