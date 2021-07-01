#%%

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('in_file',type=str, help='an integer for the accumulator')
parser.add_argument('out_file',type=str, help='an integer for the accumulator')

args = parser.parse_args()

file_name = args.in_file
out_file = args.out_file

print(args.in_file)

#%%

import zarr
import numpy as np

if file_name is None:
    file_name = "out.zarr"
if out_file is None:
    out_file = "out.png"
# zarr.load('out.zarr')
# arr = zarr.load('out.zarr')

arr = zarr.load(file_name)

# %%
from PIL import Image
im = Image.fromarray((arr*255).astype(np.uint8))
#%%
size = (np.array([int(im.height),int(im.width)])*0.5).astype(int)
im = im.resize(size, Image.ANTIALIAS)
im.save(out_file)
# %%
