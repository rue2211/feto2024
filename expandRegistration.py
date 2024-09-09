"""
Date:August 24th 2024
Author: Rudrapriya Padmanabhan
Expand feature based registration to test for a variety of frames, testing registration approach
Not the final one but to explian the potential design prior mosaicking
SIFT + AFFINE + RANSAC + KEYFRAME ADJUSTMENT
"""

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def load_images_from_folder(folder, n, file_extension="png"):
    """Load and sort images from a folder based on filenames."""
    image_files = sorted(glob.glob(os.path.join(folder, f"*.{file_extension}")))
    images = [cv2.imread(file) for file in image_files[:n]]
    return images, image_files[:n]

def bundle_adjustment(points_list, transformations, ref_points):
    """Optimize transformations to minimize local error (Affine)."""
    def residuals(params, points_list, ref_points):
        num_transforms = len(params) // 6
        residuals = []
        for i in range(num_transforms):
            M = params[i * 6:(i + 1) * 6].reshape(2, 3)
            transformed_points = np.dot(points_list[i], M[:, :2].T) + M[:, 2]

            # Ensure that only matched points are used
            min_len = min(len(transformed_points), len(ref_points))
            residuals.append(transformed_points[:min_len] - ref_points[:min_len])
        return np.concatenate(residuals).flatten()

    # Flatten the initial transformations into a single parameter array
    initial_params = np.hstack([M.flatten() for M in transformations])
    
    # Perform optimization using least squares
    optimized_params = least_squares(residuals, initial_params, args=(points_list, ref_points))

    # Reshape the optimized parameters back into affine transformation matrices
    optimized_transformations = [optimized_params.x[i * 6:(i + 1) * 6].reshape(2, 3)
                                 for i in range(len(transformations))]
    return optimized_transformations

def register_images_affine_in_segments(conf_maps, orig_images, keyframe_interval=10, roi=None):
    """Register images using separate strategies for different segments."""
    num_images = len(conf_maps)
    registered_images = []

    for start_idx in range(0, num_images, keyframe_interval):
        end_idx = min(start_idx + keyframe_interval, num_images)
        segment_conf_maps = conf_maps[start_idx:end_idx]
        segment_orig_images = orig_images[start_idx:end_idx]

        if start_idx == 0:
            # Use a more stable approach for the first segment
            registered_segment = register_images_affine(segment_conf_maps, segment_orig_images, global_reference_idx=0,roi=roi)
        else:
            # For subsequent segments, introduce recalibration and bundle adjustment
            registered_segment = register_images_affine_with_keyframes(segment_conf_maps, segment_orig_images, global_reference_idx=0, keyframe_interval=keyframe_interval,roi=roi)

        if start_idx > 0:
            # Ensure smooth transition between segments
            registered_images.extend(registered_segment[1:])
        else:
            registered_images.extend(registered_segment)

    return registered_images

def apply_roi(image, roi):
    """Apply a circular ROI to the image."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (roi['center_x'], roi['center_y']), roi['diameter'] // 2, 255, thickness=-1)
    return cv2.bitwise_and(image, image, mask=mask)

def register_images_affine(conf_maps, orig_images, global_reference_idx=0, roi=None):
    """Register the original images using a global reference frame and affine transformations."""
    key_frame = orig_images[global_reference_idx]
    registered_images = [key_frame]
    gray_conf_ref = cv2.cvtColor(conf_maps[global_reference_idx], cv2.COLOR_BGR2GRAY)
    height, width, channels = key_frame.shape
    
    sift = cv2.SIFT_create(nfeatures=5000)  # Increase the number of keypoints detected
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # Apply ROI to the key frame
    if roi:
        gray_conf_ref = apply_roi(gray_conf_ref, roi)
    
    # Detect keypoints and descriptors in the reference frame
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_conf_ref, None)
    
    for i in range(1, len(conf_maps)):
        gray_conf = cv2.cvtColor(conf_maps[i], cv2.COLOR_BGR2GRAY)
        
        # Apply ROI to the current frame
        if roi:
            gray_conf = apply_roi(gray_conf, roi)
        
        keypoints, descriptors = sift.detectAndCompute(gray_conf, None)
        
        # Apply Lowe's ratio test
        matches = bf.knnMatch(descriptors_ref, descriptors, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:  # Stricter threshold
                good_matches.append(m)
        
        if len(good_matches) > 4:
            points_ref = np.zeros((len(good_matches), 2), dtype=np.float32)
            points = np.zeros((len(good_matches), 2), dtype=np.float32)
            
            for j, match in enumerate(good_matches):
                points_ref[j, :] = keypoints_ref[match.queryIdx].pt
                points[j, :] = keypoints[match.trainIdx].pt
            
            # Estimate an affine transformation (translation + rotation + scale + shear)
            M, inliers = cv2.estimateAffine2D(points, points_ref, method=cv2.RANSAC, ransacReprojThreshold=4.0)

            
            if M is not None:
                aligned_image = cv2.warpAffine(orig_images[i], M, (width, height))
                registered_images.append(aligned_image)
            else:
                print(f"Affine transformation failed for image {i}. Skipping...")
        else:
            print(f"Not enough matches found for image {i}. Skipping...")
    
    return registered_images

def register_images_affine_with_keyframes(conf_maps, orig_images, global_reference_idx=0, keyframe_interval=15, roi=None):
    """Register images using key frames and affine transformations with local bundle adjustment."""
    registered_images = []
    transformations = []
    points_list = []

    key_frame = orig_images[global_reference_idx]
    registered_images.append(key_frame)

    # Initialize the first keyframe reference points
    gray_conf_ref = cv2.cvtColor(conf_maps[global_reference_idx], cv2.COLOR_BGR2GRAY)
    
    # Apply ROI to the key frame
    if roi:
        gray_conf_ref = apply_roi(gray_conf_ref, roi)
    
    sift = cv2.SIFT_create(nfeatures=5000)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_conf_ref, None)
    ref_points = np.array([kp.pt for kp in keypoints_ref], dtype=np.float32)

    for i in range(1, len(conf_maps)):
        gray_conf = cv2.cvtColor(conf_maps[i], cv2.COLOR_BGR2GRAY)

        # Apply ROI to the current frame
        if roi:
            gray_conf = apply_roi(gray_conf, roi)

        keypoints, descriptors = sift.detectAndCompute(gray_conf, None)
        matches = bf.knnMatch(descriptors_ref, descriptors, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 4:
            points_ref = np.zeros((len(good_matches), 2), dtype=np.float32)
            points_img = np.zeros((len(good_matches), 2), dtype=np.float32)

            for j, match in enumerate(good_matches):
                points_ref[j, :] = keypoints_ref[match.queryIdx].pt
                points_img[j, :] = keypoints[match.trainIdx].pt

            # Estimate an affine transformation
            M, inliers = cv2.estimateAffine2D(points_img, points_ref, method=cv2.RANSAC, ransacReprojThreshold=3.0)

            if M is not None:
                aligned_image = cv2.warpAffine(orig_images[i], M, (key_frame.shape[1], key_frame.shape[0]))
                registered_images.append(aligned_image)
                transformations.append(M)
                points_list.append(points_img)

                # Perform local bundle adjustment and update keyframe periodically
                if i % keyframe_interval == 0:
                    optimized_transforms = bundle_adjustment(points_list, transformations, ref_points)
                    for k in range(len(optimized_transforms)):
                        registered_images[k + 1] = cv2.warpAffine(registered_images[k + 1], optimized_transforms[k], (key_frame.shape[1], key_frame.shape[0]))
                    
                    # Update key frame and reset the points and transformations
                    key_frame = registered_images[-1]
                    keypoints_ref, descriptors_ref = sift.detectAndCompute(cv2.cvtColor(conf_maps[i], cv2.COLOR_BGR2GRAY), None)
                    ref_points = np.array([kp.pt for kp in keypoints_ref], dtype=np.float32)
                    transformations = []
                    points_list = []

            else:
                print(f"Affine transformation failed for image {i}. Skipping...")
        else:
            print(f"Not enough matches found for image {i}. Skipping...")

    return registered_images

def display_images(images, titles=None):
    """Display each image in the list individually with optional titles."""
    for i, img in enumerate(images):
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles:
            plt.title(titles[i])
        plt.axis('off')
        plt.show()

# Example usage:
confidence_maps_dir = '100/confidence_map'
original_images_dir = '100/images'
N = 100 # Number of images to process

roi_params = {
    'diameter': 389,  # Diameter of the circular ROI
    'center_x': 448 // 2,  # X position of the circle's center
    'center_y': 448 // 2,  # Y position of the circle's center
}

# Load and sort images
conf_maps, conf_files = load_images_from_folder(confidence_maps_dir, N)
orig_images, orig_files = load_images_from_folder(original_images_dir, N)

# Register images using segmented approach with ROI
registered_images = register_images_affine_in_segments(conf_maps, orig_images, keyframe_interval=10, roi=roi_params)

# Display results
display_images(registered_images, titles=[os.path.basename(f) for f in orig_files])
