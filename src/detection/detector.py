import matplotlib.pyplot as plt
import skimage
import skimage.color
import skimage.transform
import skimage.filters
import skimage.morphology
import skimage.segmentation
import numpy as np

import json
import pickle

from src.detection.EquationElement import EquationElement

class Detector:
    """
    Object to detect and classify equation elmement on a frame.
    """
    def __init__(self, frame, digit_model_path, operator_model_path, color_model_path):
        self.frame = frame
        self.element_list = []

        # load the models
        # Color KMeans
        with open(color_model_path, 'r') as f:
            self.color_clusters = json.load(f)

        # Operators K-NN
        with open(operator_model_path, 'rb') as f:
            model = pickle.load(f)
            self.operator_classifier = model['model']
            self.operator_labels_name = model['labels_name']

        # Digit MLP
        with open(digit_model_path, 'rb') as f:
            self.digit_classifier = pickle.load(f)

    def analyse_frame(self):
        """
        Analyse the frame to extract equation element and identify them.
        """
        mask = self.get_mask()
        self.extract_equation_element(mask)
        # classify element if they are op or digit
        self.classify_colors()
        # keep only digit and operators elements (not arrow)
        self.element_list = [elem for elem in self.element_list if elem.type in ['operator', 'digit']]
        # Classify digit element values
        self.classify_digits()
        # Classify operator element values
        self.classify_operators()

        return self.element_list

    def get_mask(self):
        """
        produce a binary mask of the objects.
        """
        # convert to grayscale
        img = skimage.color.rgb2gray(self.frame)
        # apply Otsu thresholding method
        mask = np.where(img < skimage.filters.threshold_otsu(img), True, False)
        # Apply some morphologycal operations to clean and connect the objects components
        mask = skimage.morphology.opening(mask, selem=skimage.morphology.disk(1))
        mask = skimage.morphology.dilation(mask, selem=skimage.morphology.disk(5))
        # complete background by region growing on the corners
        mask = skimage.segmentation.flood_fill(mask, (0,0), 0)
        mask = skimage.segmentation.flood_fill(mask, (0,mask.shape[1]-1), 0)
        mask = skimage.segmentation.flood_fill(mask, (mask.shape[0]-1,0), 0)
        mask = skimage.segmentation.flood_fill(mask, (mask.shape[0]-1,mask.shape[1]-1), 0)

        return mask

    def extract_equation_element(self, mask):
        """
        Extract the equation elements from the mask.
        """
        # labellise mask and generate the properties
        labels = skimage.measure.label(mask, background=False)
        props = skimage.measure.regionprops(labels)

        for prop in props:
            # get bounding box
            y0, x0, y1, x1 = prop.bbox
            # extract image
            elem_im = self.frame[y0:y1+1, x0:x1+1, :]
            # grayscale the image and remake the mask without the dilation
            elem_im_gray = skimage.color.rgb2gray(elem_im)
            elem_mask = np.where(elem_im_gray < skimage.filters.threshold_otsu(skimage.color.rgb2gray(elem_im_gray)), True, False)

            # reajust bbox and image to new mask
            coords = np.argwhere(elem_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            y1 = y0 + y_max+1
            x1 = x0 + x_max+1
            y0 += y_min-1
            x0 += x_min-1

            # crop to content
            elem_im = elem_im[y_min-1:y_max+2, x_min-1:x_max+2, :]
            elem_mask = elem_mask[y_min-1:y_max+2, x_min-1:x_max+2]

            # mask EquationElements
            self.element_list.append(EquationElement(elem_mask, elem_im, (x0, y0, x1, y1)))

    def classify_colors(self):
        """
        Classify the type of each equation element as the nearest neighbor of the KMeans centers.
        """
        for elem in self.element_list:
            # element color = mean over pixel of the mask
            color = np.mean(elem.image.reshape(-1,3)[elem.mask.reshape(-1)], axis=0)
            # type is the one of closest center
            names, centers = list(self.color_clusters.keys()), np.array(list(self.color_clusters.values()))
            idx = np.linalg.norm(centers - color, ord=2, axis=1).argmin()
            elem.type = names[idx]

    def classify_digits(self):
        """
        MLP on fourier descriptors
        --> evaluation
        """
        for elem in self.element_list:
            if elem.type == 'digit':
                # get image as MNIST-like
                img_mnist = self.img_as_MNIST(elem.image, elem.mask)
                # predict
                digit_pred = self.digit_classifier.predict(img_mnist.reshape(1, 28*28))
                # assign label name
                elem.value = str(digit_pred.item())

    def img_as_MNIST(self, img, mask):
        """
        Transform the digit image so that it looks like a MNIST image in term of
        range ang shape.
        """
        # put image in grayscale
        img = skimage.color.rgb2gray(img)
        # resize max_axis to 28
        img = self._resize_max(img, 28)
        mask = self._resize_max(mask, 28)
        # pad to 28,28
        h, w = img.shape
        pad_h = (int(np.ceil((28-h)/2)), int(np.floor((28-h)/2)))
        pad_w = (int(np.ceil((28-w)/2)), int(np.floor((28-w)/2)))
        img = skimage.util.pad(img, (pad_h, pad_w), constant_values=0)
        mask = skimage.util.pad(mask, (pad_h, pad_w), constant_values=0)

        # inverse colorspace and mask image
        img_masked = (255 - skimage.img_as_ubyte(img)) * skimage.img_as_bool(mask)

        # contrast stretch of images --> saturate upper 1% of pixel
        img_masked = skimage.exposure.rescale_intensity(img_masked,
                                            in_range=(0, np.percentile(img_masked, 99)),
                                            out_range=(0,255))

        return img_masked

    def _resize_max(self, img, max_len):
        """
        Resize the passed image so that it's major axis is max_len.
        """
        # take first dims
        s = img.shape
        if s[0] != s[1]:
            max_dim, min_dim = np.argmax(s), np.argmin(s)
        else:
            max_dim, min_dim = 0, 1
        aspect_ratio = s[max_dim]/s[min_dim]
        new_s = list(s)
        new_s[max_dim], new_s[min_dim] = max_len, int(max_len/aspect_ratio)
        img = skimage.transform.resize(img, new_s)

        return img

    def classify_operators(self):
        """
        5-NN on fourier descriptors for each element categorized as operators.
        """
        char_table = {'plus': '+', 'minus':'-', 'mult':'*', 'div':'/', 'eq':'='}
        for elem in self.element_list:
            if elem.type == 'operator':
                # get fourier descriptors
                feature = elem.get_Fourier_descr(5)
                # predict
                pred = self.operator_classifier.predict(feature.reshape(1, -1))
                # assign label name
                op_name = self.operator_labels_name[pred[0]]
                elem.value = char_table[op_name]
