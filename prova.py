import os
import json
import numpy as np
import pandas as pd
import functools as ft
import tensorflow as tf

from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor

from zetastitcher import InputFile

@tf.function(reduce_retracing=True)
def get_input_tf(input_file, **kwargs):
    def get_input_wrap(input_file):
        input_file = Path(input_file.decode())
        print(input_file)

        input_image = InputFile(input_file)
        print(input_image)
        input_image = input_image.astype(np.float32)

        # ATTN!: hard coded slice norm
        try:
            if kwargs["slice_p"] is not None:
                sl = int(input_file.name.split("_")[1])
                sl_p = int(kwargs["slice_p"])

                print(f"Using percentiles_{sl_p:02}.json for slice norm")
                lomin, himax = json.load(
                    open(
                        f"/home/Data/I48_slab{sl:02}/NeuN638/percentiles_{sl_p:02}.json"
                    )
                ).values()
                input_image = np.where(input_image > himax, himax, input_image)
                input_image = input_image - lomin
                input_image = np.where(input_image < 0, 0, input_image)
                input_image = input_image / (himax - lomin)
        except:
            pass

        return input_image

    input = tf.numpy_function(get_input_wrap, [input_file], tf.float32)
    return input
"""
path = "/home/amarinai/Data/I45_slice10_fused.ome.tif"
a = get_input_tf(path)
print(a.shape)
import tifffile as tiff

# Open the OME-TIFF file
with tiff.TiffFile('/home/amarinai/Data/I45_slice10_fused.ome.tif') as tif:
    # Print OME metadata
    print(tif.ome_metadata)

    # Access a specific image slice or cube
    # This example accesses the first image in the file (adjust index as needed)
    image = tif.asarray(key=206)
    print(image[10:300,10:300,...].shape) 

"""
import pathlib
from pyometiff import OMETIFFReader
path = pathlib.Path('/home/amarinai/Data/I45_slice10_fused.ome.tif')
reader =OMETIFFReader(fpath=path)
img, metadata, xml =reader.read()
print(img.shape)
print(metadata)
