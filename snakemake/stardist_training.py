from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
#matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

np.random.seed(42)
lbl_cmap = random_label_cmap()

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
# %%
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--images_dir','--images_dir',type=str, help='images_dir')
parser.add_argument('--masks_dir','--masks_dir',type=str, help='masks_dir')
parser.add_argument('--ext','--ext',default=".tif",type=str, help='ext')
parser.add_argument('--model_path','--model_path',default="models",type=str, help='model')
parser.add_argument('--epochs','--epochs',default=1,type=int)

args = parser.parse_args()

images_dir = args.images_dir
masks_dir = args.masks_dir
ext = args.ext
model_path = args.model_path
epochs = args.epochs

print(args)
# %%
# %% [markdown]
# # Data
# 
# We assume that data has already been downloaded via notebook [1_data.ipynb](1_data.ipynb).  
# 
# <div class="alert alert-block alert-info">
# Training data (for input `X` with associated label masks `Y`) can be provided via lists of numpy arrays, where each image can have a different size. Alternatively, a single numpy array can also be used if all images have the same size. Label images need to be integer-valued.
# </div>

# %%

images = f"{images_dir}/*{ext}"
masks = f"{masks_dir}/*{ext}"

X = sorted(glob(images))
Y = sorted(glob(masks))
print(len(X))
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

# %%
X = list(map(imread,X))
Y = list(map(imread,Y))
print(X)
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

# %% [markdown]
# Normalize images and fill small label holes.

# %%
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

# %% [markdown]
# Split into train and validation datasets.

# %%
assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

# %% [markdown]
# Training data consists of pairs of input image and label instances.

# %%
def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl, cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()

# %%
i = min(9, len(X)-1)
img, lbl = X[i], Y[i]
assert img.ndim in (2,3)
img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
plot_img_label(img,lbl)
None;

# %% [markdown]
# # Configuration
# 
# A `SplineDist2D` model is specified via a `Config2D` object.

# %%
print(Config2D.__doc__)

# %%
# choose the number of control points (M)
# 32 is a good default choice (see 1_data.ipynb)
n_rays = 32

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = False and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2,2)

conf = Config2D (
    n_rays       = n_rays,
    grid         = grid,
    use_gpu      = use_gpu,
    n_channel_in = n_channel,
)
print(conf)
vars(conf)

# from stardist.models import StarDist2D

# %%

model = StarDist2D(conf, name='stardist', basedir=model_path)
# model = StarDist2D.from_pretrained("2D_paper_dsb2018",name='stardist', basedir=model_path)

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

print(dir(model))
# %% [markdown]
# Check if the neural network has a large enough field of view to see up to the boundary of most objects.

# %%
# median_size = calculate_extents(list(Y), np.median)
# fov = np.array(model._axes_tile_overlap('YX'))
# print(f"median object size:      {median_size}")
# print(f"network field of view :  {fov}")
# if any(median_size > fov):
#     print("WARNING: median object size larger than field of view of the neural network.")

# %% [markdown]
# # Data Augmentation

# %% [markdown]
# You can define a function/callable that applies augmentation to each batch of the data generator.  
# We here use an `augmenter` that applies random rotations, flips, and intensity changes, which are typically sensible for (2D) microscopy images (but you can disable augmentation by setting `augmenter = None`).

# %%
def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

# %%
# plot some augmented examples
img, lbl = X[0],Y[0]
plot_img_label(img, lbl)
for _ in range(3):
    img_aug, lbl_aug = augmenter(img,lbl)
    plot_img_label(img_aug, lbl_aug, img_title="image augmented", lbl_title="label augmented")

# %% [markdown]
# # Training

# %%
model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter, epochs = epochs)
# model.save(model_path)
# %% [markdown]
# # Visualization

# %% [markdown]
# First predict the labels for all validation images:

# # %%
# Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
#               for x in tqdm(X_val)]

# # %% [markdown]
# # Plot a GT/prediction example 

# # %%
# plot_img_label(X_val[0],Y_val[0], lbl_title="label GT")
# plot_img_label(X_val[0],Y_val_pred[0], lbl_title="label Pred")


