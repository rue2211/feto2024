"""
Author: Rudrapriya Padmanabhan

First iteration of the mosaicking script, included feature based registration and plots the mosaic onto a canvas
"""

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def load_images_from_folder(folder, n, file_extension="png"):
    """Load and sort images from a folder based on filenames."""
    image_files = sorted(glob.glob(os.path.join(folder, f"*.{file_extension}")))
    images = [cv2.imread(file) for file in image_files[:n]]
    return images, image_files[:n]

def register_images_affine(conf_maps, orig_images):
    """Register images and return the cumulative transformations."""
    registered_images = []
    transformations = []
    
    # Initialize identity matrix for the first image
    cumulative_transform = np.eye(3)

    registered_images.append(orig_images[0])
    transformations.append(cumulative_transform)

    # SIFT and BFMatcher initialization
    sift = cv2.SIFT_create(nfeatures=5000)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Convert first confidence map to grayscale
    gray_conf_ref = cv2.cvtColor(conf_maps[0], cv2.COLOR_BGR2GRAY)
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_conf_ref, None)

    for i in range(1, len(conf_maps)):
        gray_conf = cv2.cvtColor(conf_maps[i], cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_conf, None)

        matches = bf.knnMatch(descriptors_ref, descriptors, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(good_matches) > 4:
            points_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches])
            points_img = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

            M, _ = cv2.estimateAffine2D(points_img, points_ref, method=cv2.RANSAC, ransacReprojThreshold=4.0)

            if M is not None:
                M_homogeneous = np.vstack([M, [0, 0, 1]])  # Convert to homogeneous coordinates
                cumulative_transform = cumulative_transform @ M_homogeneous

                registered_images.append(orig_images[i])
                transformations.append(cumulative_transform)
                keypoints_ref, descriptors_ref = keypoints, descriptors  # Update reference
            else:
                print(f"Affine transformation failed for image {i}. Skipping...")
        else:
            print(f"Not enough matches found for image {i}. Skipping...")

    return registered_images, transformations

def create_mosaic(registered_images, transformations, canvas_size=(2000, 2000)):
    """Create a mosaic from the list of registered images using their transformations."""
    # Initialize a blank canvas
    mosaic = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    
    # Set an offset to place the first image at the center of the canvas
    offset_x, offset_y = canvas_size[0] // 2, canvas_size[1] // 2

    for i, (img, transform) in enumerate(zip(registered_images, transformations)):
        # Calculate the transformed coordinates
        transform_with_offset = np.array(transform)
        transform_with_offset[:2, 2] += [offset_x, offset_y]
        
        # Warp the image based on the accumulated transformation
        warped_img = cv2.warpPerspective(img, transform_with_offset, (canvas_size[0], canvas_size[1]))

        # Blend the current warped image onto the mosaic
        mask = np.any(warped_img > 0, axis=2).astype(np.uint8) * 255
        mosaic = cv2.bitwise_and(mosaic, mosaic, mask=cv2.bitwise_not(mask))
        mosaic = cv2.add(mosaic, warped_img)

    return mosaic

def display_images(images, titles=None):
    """Display each image in the list individually with optional titles."""
    for i, img in enumerate(images):
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles:
            plt.title(titles[i])
        plt.axis('off')
        plt.show()


confidence_maps_dir = 'results/video1308/Mosaic_To_Process/undistorted/conf_maps'
original_images_dir = 'results/video1308/Mosaic_To_Process/undistorted/images'
N = 120

# Load and sort images
conf_maps, conf_files = load_images_from_folder(confidence_maps_dir, N)
orig_images, orig_files = load_images_from_folder(original_images_dir, N)

# Register images
registered_images, transformations = register_images_affine(conf_maps, orig_images)

# Create and display the mosaic
mosaic = create_mosaic(registered_images, transformations)
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Mosaic Image")
plt.show()
