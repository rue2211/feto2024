"""
Date: August 15th 2024
Author: Rudrapriya Padmanabhan

Removing distortion from the images and confidence maps sent to segmentation model
As the frames are now cropped and resized to 448x448 the intrinsic values would be modified to reflect the new principle point
Distortion removed basis the ROI of the image
"""

import cv2
import numpy as np
import os

def read_camera_parameters(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove brackets and split each line correctly
    camera_matrix = np.array([
        [float(val) for val in lines[2].strip().replace('[', '').replace(']', '').split()],
        [float(val) for val in lines[3].strip().replace('[', '').replace(']', '').split()],
        [float(val) for val in lines[4].strip().replace('[', '').replace(']', '').split()]
    ])

    # Remove brackets and split for distortion coefficients
    dist_coeffs = np.array([float(val) for val in lines[6].strip().replace('[', '').replace(']', '').split()])

    return camera_matrix, dist_coeffs

def apply_circular_roi(image, diameter):
    """
    Apply a circular mask to the image and crop it to the circular ROI.
    The mask is centered in the middle of the image and has the specified diameter.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    radius = diameter // 2

    # Create a circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Crop to the bounding box of the circle
    x_start = center[0] - radius
    y_start = center[1] - radius
    cropped_image = masked_image[y_start:y_start + diameter, x_start:x_start + diameter]

    return cropped_image

def adjust_intrinsics_for_crop(camera_matrix, x_crop, y_crop):
    """
    Adjust the camera matrix (intrinsic parameters) based on the crop offset.
    """
    adjusted_matrix = camera_matrix.copy()
    
    # Adjust the principal point (cx, cy)
    adjusted_matrix[0, 2] -= x_crop  # Adjust cx
    adjusted_matrix[1, 2] -= y_crop  # Adjust cy

    return adjusted_matrix

def undistort_and_resize_image(image_path, camera_matrix, dist_coeffs, x_crop, y_crop, output_path, target_size=(448, 448), roi_diameter=396):
    image = cv2.imread(image_path)
    
    # Apply circular ROI before undistortion
    roi_image = apply_circular_roi(image, roi_diameter)
    
    # Adjust the intrinsic matrix for the crop
    adjusted_camera_matrix = adjust_intrinsics_for_crop(camera_matrix, x_crop, y_crop)
    
    # Obtain the optimal new camera matrix
    height, width = roi_image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(adjusted_camera_matrix, dist_coeffs, (width, height), 1, (width, height))

    # Undistort the image
    undistorted_image = cv2.undistort(roi_image, adjusted_camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Resize the undistorted image
    resized_image = cv2.resize(undistorted_image, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Save the final resized image
    cv2.imwrite(output_path, resized_image)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_folder(images_dir, conf_maps_dir, output_images_dir, output_conf_maps_dir, camera_matrix, dist_coeffs, x_crop, y_crop, target_size=(448, 448), roi_diameter=396):
    # Create directories for final outputs
    create_directory(output_images_dir)
    create_directory(output_conf_maps_dir)
    
    # Process each image and its corresponding confidence map
    for filename in os.listdir(images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_dir, filename)
            conf_map_path = os.path.join(conf_maps_dir, filename)
            
            # Define output paths
            final_image_output = os.path.join(output_images_dir, f'final_{filename}')
            final_conf_map_output = os.path.join(output_conf_maps_dir, f'final_conf_{filename}')
            
            # Process the image
            undistort_and_resize_image(image_path, camera_matrix, dist_coeffs, x_crop, y_crop, final_image_output, target_size, roi_diameter)
            
            # Process the corresponding confidence map (if exists)
            if os.path.exists(conf_map_path):
                undistort_and_resize_image(conf_map_path, camera_matrix, dist_coeffs, x_crop, y_crop, final_conf_map_output, target_size, roi_diameter)

# Example usage:
camera_parameters_file = 'results/video1308/acquisition/intriniscs_refinement/intrinsics.txt'
camera_matrix, dist_coeffs = read_camera_parameters(camera_parameters_file)

# New closer crop offsets
x_crop = 610  # Horizontal offset of the new closer crop
y_crop = 192  # Vertical offset of the new closer crop

# Input directories
images_dir = 'results/video1308/Mosaic_To_Process/images'
conf_maps_dir = 'results/video1308/Mosaic_To_Process/predicted_mask'

# Output directories
output_images_dir = 'results/video1308/Mosaic_To_Process/undistorted/images'
output_conf_maps_dir = 'results/video1308/Mosaic_To_Process/undistorted/conf_maps'

# Process the folders with a circular ROI
process_folder(images_dir, conf_maps_dir, output_images_dir, output_conf_maps_dir, camera_matrix, dist_coeffs, x_crop, y_crop, roi_diameter=370)

