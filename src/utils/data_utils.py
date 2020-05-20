import gzip
import numpy as np
import skimage.morphology
import skimage.filters
import skimage.io
import glob

from src.detection.EquationElement import EquationElement

def get_operators_train_data(data_path, Nfeat=4, rotate=True):
    """
    Provide the operators train features (The first Nfeat Fourier descriptor).
    If rotate is True, each image is rotated to increase the diversity of images.
    """
    # Read images and labels
    mask, label = [], []
    for mask_fn in glob.glob(data_path+'mask/*.png'):
        mask.append(skimage.img_as_bool(skimage.io.imread(mask_fn)))
        label.append(mask_fn.split('/')[-1].split('_')[-1][:-4])

    label_correspondance = {i: name for i, name in zip(range(len(label)), label)}

    # Rotate images
    if rotate:
        mask_rot = []
        for m in mask:
            mask_rot += [skimage.transform.rotate(m, ang, order=0, resize=True) for ang in np.arange(0,360,1)]
        mask = mask_rot
        # expand labels
        labels = np.repeat(np.array(range(len(label))), 360)

    # get Fourier Fetaures as numpy array
    feat = np.stack([EquationElement(m, None).get_Fourier_descr(Nfeat) for m in mask], axis=0)

    return feat, labels, label_correspondance

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
    Extract MNIST labels as a Numpy array from the byte file.
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def get_MNIST(data_path, digits=[0,1,2,3,4,5,6,7,8,9], image_shape=(28,28),
              train_size=60000, test_size=10000, binary=False, add_rotation=0):
    """
    Get the MNIST dataset for the required digits from the provided data folder
    path. Binarize the image if required. Add x rotated version of each image to
    the sets.
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

    # add rotated version of images
    if add_rotation > 0:
        train_images_rot = [train_images]
        test_images_rot = [test_images]
        for _ in range(add_rotation):
            train_images_rot.append(np.stack([skimage.transform.rotate(im, np.random.randint(0,360), order=1) for im in train_images], axis=0))
            test_images_rot.append(np.stack([skimage.transform.rotate(im, np.random.randint(0,360), order=1) for im in test_images], axis=0))

        train_images = np.concatenate(train_images_rot, axis=0)
        train_labels = np.concatenate([train_labels]*(add_rotation+1), axis=0)
        test_images = np.concatenate(test_images_rot, axis=0)
        test_labels = np.concatenate([test_labels]*(add_rotation+1), axis=0)

    return train_images, train_labels, test_images, test_labels
