# %%
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

np.random.seed(6)
import os

lbl_cmap = random_label_cmap()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %%
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--image_in','--image_in',type=str, help='images_dir')
parser.add_argument('--figure_out','--figure_out',type=str, help='figure_out')
parser.add_argument('--model_path','--model_path',default="models",type=str, help='model')


args = parser.parse_args()

image_in = args.image_in
model_path = args.model_path
figure_out = args.figure_out

print(args)

X = [image_in]
X = list(map(imread,X))
plt.imshow(X[0])
plt.show()

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

model = StarDist2D(None, name='stardist', basedir='models')

img = normalize(X[0], 1,99.8, axis=axis_norm)
labels, details = model.predict_instances(img)

plt.figure(figsize=(8,8))
plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
plt.axis('off')
plt.savefig(figure_out)



