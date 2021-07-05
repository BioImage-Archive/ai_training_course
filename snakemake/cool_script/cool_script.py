#%%

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('in_file',type=str, help='Input file')
parser.add_argument('out_file',type=str, help='Input file')

args = parser.parse_args()

file_name = args.in_file
out_file = args.out_file

print(args.in_file)

#%%
import pims
import numpy as np

if file_name is None:
    file_name = "Stardist/Test - Images/cell migration R1 - Position 58_XY1562686154_Z0_T00_C1-image76.tif"
if out_file is None:
    out_file = "out.zarr"

image = pims.open(file_name)
image_np = np.array(pims.open(file_name)[0])

# %%
from sklearn.preprocessing import minmax_scale
image_np_base_correct = image_np-image_np.min()
image_np_scaled = image_np_base_correct/image_np_base_correct.max()

# scaled = minmax_scale(image_np,feature_range=(0,1))
# %%

# image_scaled = pims.Image(image_np_scaled[0])
# %%
# pims(image_np_scaled)
# %%
import zarr

zarr.save(out_file, image_np_scaled)

# zarr.load('out.zarr')

# from ome_zarr.writer import write_image

# # %%
# import pathlib
# # from ome_zarr.io import parse_url
# # store = zarr.DirectoryStore('data/example.zarr')
# root = zarr.open('out.zarr', mode='w')
# # path = pathlib.Path(tmpdir.mkdir("data"))
# # store = parse_url(path, mode="w").store
# # root = zarr.group(store=store)
# # group = root.create_group("test")
# # write_image(image_np_scaled,)

# %%
