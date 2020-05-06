import click
import sys
import os
import glob
import json

import skimage.io
import numpy as np
from sklearn.cluster import KMeans

def color_code(rgb):
    """

    """
    r05 = int(rgb[0] / 256 * 5)
    g05 = int(rgb[1] / 256 * 5)
    b05 = int(rgb[2] / 256 * 5)

    c_idx = 16 + 36 * r05 + 6 * g05 + b05

    return f"\x1b[48;5;255m" + f"\x1b[38;5;{c_idx}m"
END = "\x1b[0m"

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
        cluster_names.append(click.prompt(f'|---- Please enter a name for the cluser coresponding to the RGB code '+color_code(c)+f'{c}'+END, type=str))

    # make output and save
    output = {c_name:c.tolist() for c, c_name in zip(centers, cluster_names)}
    with open(model_path + 'KMeanse_centers.json', 'w') as fp:
        json.dump(output, fp)

if __name__ == '__main__':
    main()
