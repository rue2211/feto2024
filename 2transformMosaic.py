"""
Test if its the individual frame transformaitons or the cumulative transformation
"""

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import csv

def load_images_from_folder(folder, n, file_extension="png"):
    #Load and sort images from a folder based on filenames.
    image_files = sorted(glob.glob(os.path.join(folder, f"*.{file_extension}")))
    images = [cv2.imread(file) for file in image_files[:n]]
    return images, image_files[:n]

def save_cumulative_transformations_to_csv(cumulative_transformations, csv_filename):
    #save cumulative transformations to a CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [
            "Image_Index", 
            "M11", "M12", "M13", 
            "M21", "M22", "M23", 
            "M31", "M32", "M33"
        ]
        writer.writerow(header)
        
        for i, cumulative_transform in enumerate(cumulative_transformations):
            row = [i] + cumulative_transform.flatten().tolist()
            writer.writerow(row)

def save_frame_transformations_to_csv(frame_transformations, csv_filename):
    #save individual frame transformations to a separate CSV file.
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [
            "Image_Index", 
            "M11", "M12", "M13", 
            "M21", "M22", "M23", 
            "M31", "M32", "M33"
        ]
        writer.writerow(header)
        
        for i, frame_transform in enumerate(frame_transformations):
            row = [i] + frame_transform.flatten().tolist()
            writer.writerow(row)

def register_images_affine(conf_maps, orig_images, cumulative_csv="cumulative_transformations.csv", frame_csv="frame_transformations.csv"):
   #function to register via affine transfroms 
    registered_images = []
    cumulative_transformations = []
    frame_transformations = []

    cumulative_transform = np.eye(3)

    registered_images.append(orig_images[0])
    #cumulative transformations load
    cumulative_transformations.append(cumulative_transform)
    frame_transformations.append(np.eye(3))
    #SIFT and brute force matched
    sift = cv2.SIFT_create(nfeatures=5000)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

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
                cumulative_transformations.append(cumulative_transform)
                frame_transformations.append(M_homogeneous)

                keypoints_ref, descriptors_ref = keypoints, descriptors  # Update reference
            else:
                print(f"Affine transformation failed for image {i}. Skipping...")
                frame_transformations.append(np.eye(3))  # Append identity transformation for skipped image
        else:
            print(f"Not enough matches found for image {i}. Skipping...")
            frame_transformations.append(np.eye(3))  # Append identity transformation for skipped image

    # Save the transformations to separate CSV files - why 2 transforms 
    save_cumulative_transformations_to_csv(cumulative_transformations, cumulative_csv)
    save_frame_transformations_to_csv(frame_transformations, frame_csv)
    
    return registered_images, cumulative_transformations, frame_transformations

def create_mosaic(registered_images, transformations, canvas_size=(2000, 1500)):
    """Create a mosaic from the list of registered images using their transformations."""
    mosaic = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    
    offset_x, offset_y = canvas_size[0] // 2, canvas_size[1] // 2

    for i, (img, transform) in enumerate(zip(registered_images, transformations)):
        transform_with_offset = np.array(transform)
        transform_with_offset[:2, 2] += [offset_x, offset_y]
        
        warped_img = cv2.warpPerspective(img, transform_with_offset, (canvas_size[0], canvas_size[1]))

        mask = np.any(warped_img > 0, axis=2).astype(np.uint8) * 255
        smoothed_mask = cv2.GaussianBlur(mask, (3, 3), 0)

        mosaic = cv2.bitwise_and(mosaic, mosaic, mask=cv2.bitwise_not(smoothed_mask))
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


confidence_maps_dir = '100/confidence_map'
original_images_dir = '100/images'
N = 101 #size of mosaic for testing 

# load data
conf_maps, conf_files = load_images_from_folder(confidence_maps_dir, N)
orig_images, orig_files = load_images_from_folder(original_images_dir, N)

# save data
registered_images, cumulative_transformations, frame_transformations = register_images_affine(
    conf_maps, orig_images, 
    cumulative_csv="cumulative_transformations.csv", 
    frame_csv="frame_transformations.csv"
)

# Plottting
mosaic = create_mosaic(registered_images, cumulative_transformations)
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Mosaic Image")
plt.savefig('100/image_mosaic.png')
plt.show()
