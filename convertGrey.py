"""
Date: July 2024
Author: Rudrapriya Padmanabhan

Script to convert RGB ground truth segmenetation into greyscale
Feed to fine tune model
"""

import os
from PIL import Image
import numpy as np

# Define the RGB to Grayscale mapping
color_to_gray = {
    (0, 0, 0): 0,       # Background -> 0
    (255, 0, 0): 1,     # Vessel -> 1
    (0, 0, 255): 2,     # Tool -> 2
    (0, 255, 0): 3      # Fetus -> 3
}

# Directories
input_directory = 'video1537/labels_rgb'  # Directory containing RGB masks
output_directory = 'video1537/labels'  # Directory to save grayscale masks

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through each file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.png'):  # Assuming masks are in PNG format
        input_image_path = os.path.join(input_directory, filename)
        output_image_path = os.path.join(output_directory, filename)
        
        # Load the RGB mask
        rgb_mask = Image.open(input_image_path)

        # Check if the image has an alpha channel (RGBA)
        if rgb_mask.mode == 'RGBA':
            # Convert the image to RGB
            rgb_mask = rgb_mask.convert('RGB')

        # Convert the image to numpy array
        rgb_array = np.array(rgb_mask)

        # Create an empty array for the grayscale mask
        gray_mask = np.zeros((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.uint8)

        # Apply the mapping
        for rgb, gray in color_to_gray.items():
            mask = np.all(rgb_array == rgb, axis=-1)
            gray_mask[mask] = gray

        # Save the grayscale mask
        gray_mask_image = Image.fromarray(gray_mask)
        gray_mask_image.save(output_image_path)

        print(f"Converted and saved: {filename}")

print("All files have been processed.")