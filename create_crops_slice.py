import pathlib
import os
import tifffile 
import pathlib
from pyometiff import OMETIFFReader
import random

# Set the desired output directory
output_dir = "/home/amarinai/Data/Random_crops_long/Tiff_files/Train" 
os.makedirs(output_dir, exist_ok=True)
output_shape = [50, 100, 100]

# Path to the TIFF OME file
tiff_ome_path = '/home/amarinai/Data/l45_unlabeled/I45_slice10_fused.ome.tif'
path = pathlib.Path(tiff_ome_path)  # '/home/amarinai/Data/I45_slice10_fused.ome.tif'

# Initialize the OME-TIFF reader and read the image
reader = OMETIFFReader(fpath=path)
print("preparing to read... this will take a few minutes")
img_to_crop, metadata, _ = reader.read()
print(img_to_crop.shape)
print(metadata)
print("entering the loop")

crops = []
for i in range(102):
    # Extract the crop from the image

    start_depth = random.randint(0, img_to_crop.shape[0] - output_shape[0])
    start_height = random.randint(0, img_to_crop.shape[1] - output_shape[1])
    start_width = random.randint(0, img_to_crop.shape[2] - output_shape[2])
    
    # Extract the random crop from the image
    crop = img_to_crop[start_depth:start_depth + output_shape[0],
                       start_height:start_height + output_shape[1],
                       start_width:start_width + output_shape[2]]    
    
    print(i)
    print(crop.shape)
    
    # Save the crop to the specified output directory
    crop_filename = os.path.join(output_dir, f"crop_{i}.tif")
    tifffile.imwrite(crop_filename, crop)  # or use Image.fromarray(crop).save(crop_filename) if using PIL
    
    # Append the crop to the list (if needed)
    crops.append(crop)

print("the loop is over..")
print(crops)
