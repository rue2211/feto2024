"""
Sanity test to see if direct registration works between two frames
"""

import cv2
import numpy as np

def register_images_with_confidence_maps(image1_path, image2_path, conf_map1_path, conf_map2_path, output_path):
    # Load the images and confidence maps
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    conf_map1 = cv2.imread(conf_map1_path, cv2.IMREAD_GRAYSCALE)
    conf_map2 = cv2.imread(conf_map2_path, cv2.IMREAD_GRAYSCALE)

    # Detect ORB features and descriptors based on confidence maps
    orb = cv2.ORB_create(5000)
    keypoints1, descriptors1 = orb.detectAndCompute(conf_map1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(conf_map2, None)

    # Match features using the Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the image to align it with the other
    height, width, channels = image2.shape
    aligned_image = cv2.warpPerspective(image1, H, (width, height))

    # Save the aligned image
    cv2.imwrite(output_path, aligned_image)

    return aligned_image, H

# Example usage:
image1_path = 'final_resized_image1.jpg'
image2_path = 'final_resized_image2.jpg'
conf_map1_path = 'final_resized_conf_map1.png'
conf_map2_path = 'final_resized_conf_map2.png'
output_path = 'aligned_image.jpg'

aligned_image, homography = register_images_with_confidence_maps(image1_path, image2_path, conf_map1_path, conf_map2_path, output_path)
