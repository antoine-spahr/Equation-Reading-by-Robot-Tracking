import click
import os
import sys
sys.path.append('../')

import imageio

from src.detection import detector

@click.commande()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('output_path', type=click.Path(exists=False), default='')
def main(video_path, output_path):
    """
    Read the equation showed on the video at `video_path` by tracking the robot.
    The video with the written equation and the robot trajectory is saved at
    `output_path`.
    """
    # laod video from video_path
    video = imageio.get_reader(video_path)

    # Read 1st frame and read the environment (classify operator and digits)
    x = video.get_data(0)

    # initialize equation <- '' and output-video <- []

    # while frame < video_length or '=' found:
    #       track robot position
    #       append position to list
    #       check if bbox more than 90% overlap with any digit/operator
    #           append character to equation string
    #           if character is '='
    #               exit tracking
    #           evaluate equation and append results to strin
    #       draw track and equation on output frame

    # if '=' not found add error message on output-video

    # save output-video at output_path

if __name__ == '__main__':
    main()


###############################################################################


import matplotlib.pyplot as plt
import skimage
import skimage.filters
import skimage.morphology
import skimage.segmentation
import numpy as np
video_path = r'../data/robot_parcours_1.avi'
video = imageio.get_reader(video_path)

img = video.get_data(0)

plt.imshow(video.get_data(0))

#%% convert rgb to grayscale
img1 = video.get_data(0)
img = skimage.color.rgb2gray(img1)
#img = skimage.filters.median(img, selem=skimage.morphology.disk(2))

# apply Otsu thresholding method
thres = skimage.filters.threshold_otsu(img)
#mask = skimage.filters.apply_hysteresis_threshold(img, low=0.8*thres, high=1*thres)
mask = np.where(img < skimage.filters.threshold_otsu(img), True, False)
#mask = skimage.morphology.closing(mask, selem=skimage.morphology.square(3))
mask = skimage.morphology.opening(mask, selem=skimage.morphology.disk(1))
mask = skimage.morphology.dilation(mask, selem=skimage.morphology.disk(5))
# complete background
mask = skimage.segmentation.flood_fill(mask, (0,0), 0)
mask = skimage.segmentation.flood_fill(mask, (0,mask.shape[1]-1), 0)
mask = skimage.segmentation.flood_fill(mask, (mask.shape[0]-1,0), 0)
mask = skimage.segmentation.flood_fill(mask, (mask.shape[0]-1,mask.shape[1]-1), 0)

fig, ax = plt.subplots(1,1,figsize=(9,9))
ax.imshow(mask, cmap='gray')
plt.show()


#%% get bbox objects and masks
labels = skimage.measure.label(mask, background=False)
props = skimage.measure.regionprops(labels, intensity_image=img)

y0, x0, y1, x1 = props[0].bbox
elem_im = img1[y0:y1+1, x0:x1+1, :]
elem_im_gray = skimage.color.rgb2gray(elem_im)
# remake the mask of object
elem_mask = np.where(elem_im_gray < skimage.filters.threshold_otsu(skimage.color.rgb2gray(elem_im_gray)), True, False)

# reajust bbox and image
coords = np.argwhere(elem_mask)
y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)

y1 = y0 + y_max+1
x1 = x0 + x_max+1
y0 += y_min-1
x0 += x_min-1

elem_im = elem_im[y_min-1:y_max+2, x_min-1:x_max+2, :]
elem_mask = elem_mask[y_min-1:y_max+2, x_min-1:x_max+2]

plt.imshow(elem_im * np.stack(3*[elem_mask], axis=2))

# %%
