import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.color
import numpy as np

class Tracker:
    """

    """
    def __init__(self):
        """

        """
        self.position_list = []
        self.bbox = None

    def get_arrow_mask(self, frame):
        """

        """
        #img = skimage.exposure.rescale_intensity(frame, in_range=tuple(np.percentile(frame, (2,98))), out_range=(0,255))
        img = skimage.exposure.equalize_adapthist(frame)
        img = skimage.color.rgb2gray(img)
        # # apply Otsu thresholding method
        thres = skimage.filters.threshold_otsu(img)
        # #mask = skimage.filters.apply_hysteresis_threshold(img, low=0.8*thres, high=1*thres)
        mask = np.where(img < skimage.filters.threshold_otsu(img), True, False)
        mask = skimage.morphology.binary_opening(mask, skimage.morphology.disk(5))
        #mask = skimage.morphology.binary_closing(mask, skimage.morphology.disk(5))
        mask = skimage.segmentation.flood_fill(mask, (0,0), 0)
        mask = skimage.segmentation.flood_fill(mask, (0,mask.shape[1]-1), 0)
        mask = skimage.segmentation.flood_fill(mask, (mask.shape[0]-1,0), 0)
        mask = skimage.segmentation.flood_fill(mask, (mask.shape[0]-1,mask.shape[1]-1), 0)

        return mask

    def track(self, frame):
        """

        """
        mask = self.get_arrow_mask(frame)
        label = skimage.measure.label(mask)
        props = skimage.measure.regionprops(label)

        arrow = max(props, key=lambda prop: prop.area)
        yc, xc = arrow.centroid
        self.position_list.append((xc, yc))
        y0, x0, y1, x1 = arrow.bbox
        self.bbox = (x0, y0, x1, y1)
