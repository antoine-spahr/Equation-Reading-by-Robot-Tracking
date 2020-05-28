import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.segmentation
import skimage.color
import numpy as np

class Tracker:
    """
    Tracker object to follow the red arrow on the video.
    """
    def __init__(self):
        """
        Initialize the tracker. Position list contains the list of tracked postion.
        bbox contains the current bounding box (x0, y0, x1, y1).
        """
        self.position_list = []
        self.bbox = None

    def get_arrow_mask(self, frame):
        """
        Detect the arrow on the passed frame.
        """
        # Adapte contrast and put in grayscale
        img = skimage.exposure.equalize_adapthist(frame)
        img = skimage.color.rgb2gray(img)
        # # apply Otsu thresholding method
        thres = skimage.filters.threshold_otsu(img)
        mask = np.where(img < skimage.filters.threshold_otsu(img), True, False)
        # Morphological cleaning
        mask = skimage.morphology.binary_opening(mask, skimage.morphology.disk(5))
        # Remove corners of room
        mask = skimage.segmentation.flood_fill(mask, (0,0), 0)
        mask = skimage.segmentation.flood_fill(mask, (0,mask.shape[1]-1), 0)
        mask = skimage.segmentation.flood_fill(mask, (mask.shape[0]-1,0), 0)
        mask = skimage.segmentation.flood_fill(mask, (mask.shape[0]-1,mask.shape[1]-1), 0)

        return mask

    def track(self, frame):
        """
        Get Centroid position and bbox of the arrow from the passed frame.
        """
        mask = self.get_arrow_mask(frame)
        label = skimage.measure.label(mask)
        props = skimage.measure.regionprops(label)

        arrow = max(props, key=lambda prop: prop.area)
        yc, xc = arrow.centroid
        self.position_list.append((xc, yc))
        y0, x0, y1, x1 = arrow.bbox
        self.bbox = (x0, y0, x1, y1)
