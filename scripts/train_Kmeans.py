import click
import sys
import os
import glob
import json

import skimage.io
import numpy as np
from sklearn.cluster import KMeans

@click.command()
@click.argument('data_folder', type=click.Path(exists=True))
@click.option('--model_path', type=click.Path(exists=False), default='', help='Where to save the model.')
@click.option('--n_clusters', type=int, default=3, help='The numbr of cluster to search.')
def main(data_folder, model_path, n_clusters):
    """
    Detect colors in image.
    """
    img, mask = [], []
    for img_fn in glob.glob(data_folder+'img/*.png'):
        img.append(skimage.io.imread(img_fn))
        mask.append(skimage.img_as_bool(skimage.io.imread(img_fn.replace('img', 'mask'))))
    # linearize and stack data
    data = np.concatenate([im.reshape(-1, 3)[msk.reshape(-1)] for im, msk in zip(img, mask)], axis=0)
    # Kmean clustering
    kmean = KMeans(n_clusters=n_clusters).fit(data)
    centers = kmean.cluster_centers_

    # ask user
    cluster_names = []
    for c in centers:
        cluster_names.append(click.prompt(f'|---- Please enter a name for the cluser coresponding to the RGB code {c}', type=str))

    # make output and save
    output = {c_name:c.tolist() for c, c_name in zip(centers, cluster_names)}
    with open(model_path + 'KMeanse_centers.json', 'w') as fp:
        json.dump(output, fp)

if __name__ == '__main__':
    main()



# data_folder = '../data/Train_objects/'
# model_path = '../models/'
# n_clusters = 3
#
# # load images and mask
# img, mask = [], []
# for img_fn in glob.glob(data_folder+'img/*.png'):
#     img.append(skimage.io.imread(img_fn))
#     mask.append(skimage.img_as_bool(skimage.io.imread(img_fn.replace('img', 'mask'))))
# # linearize and stack data
# data = np.concatenate([im.reshape(-1, 3)[msk.reshape(-1)] for im, msk in zip(img, mask)], axis=0)
# # Kmean clustering
# kmean = KMeans(n_clusters=n_clusters).fit(data)
# centers = kmean.cluster_centers_
#
# # ask user
# cluster_names = []
# for c in centers
#     cluster_names.append(click.prompt(f'|---- Please enter a name for the cluser coresponding to the RGB code {c} :', type=str))
#
#
#
# # make output and save
# output = {c_name:c.tolist() for c, c_name in zip(centers, cluster_names)}
# with open(model_path + 'KMeanse_centers.json', 'w') as fp:
#     json.dump(output, fp)
#
#
# # %%
# img_fn = glob.glob(data_folder+'img/*.png')[1]
#
# img = skimage.io.imread(img_fn)
# mask = skimage.img_as_bool(skimage.io.imread(img_fn.replace('img', 'mask')))
#
# x = np.mean(img.reshape(-1, 3)[mask.reshape(-1)], axis=0)
#
# np.argmin(np.linalg.norm(x-c, ord=2, axis=1))
#
# # %%
# img_masked = img * np.stack([mask]*3, axis=2)
# plt.imshow(img_masked)
#%%
