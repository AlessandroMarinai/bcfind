import os
import shutil

# Define the folder containing the .tiff files
folder_path = '/home/amarinai/Data/Broca_old_l48_long/Tiff_files/Train'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
        # Construct the original file path
        original_file_path = os.path.join(folder_path, filename)
        
        # Create the new filename by appending '_1' before the extension
        new_filename = filename.replace('.tif', '_1.tif')

        
        # Construct the new file path
        new_file_path = os.path.join(folder_path, new_filename)
        
        # Copy the file
        shutil.copy2(original_file_path, new_file_path)

print("Files copied successfully.")
