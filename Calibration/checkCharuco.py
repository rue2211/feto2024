"""
Debug script to see charuco - not part of project pipeline
"""

import cv2
import numpy as np
import glob

# Step 1: Set up the Charuco board detector
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
chessboard_corners = (9, 7)  
square_size_mm = 5  
marker_size_mm = 3  
board = cv2.aruco.CharucoBoard_create(chessboard_corners[0], chessboard_corners[1], square_size_mm, marker_size_mm, dictionary)

# Load intrinsic parameters from the file
def load_intrinsics(intrinsics_path):
    """
    Load the intrinsic matrix and distortion coefficients from a text file.
    
    The file format is expected to contain:
    - A line with "Intrinsics are:", followed by the 3x3 intrinsic matrix.
    - A line with "Distortion matrix is:", followed by the distortion coefficients.
    """
    with open(intrinsics_path, 'r') as f:
        lines = f.readlines()
    
    # Initialize variables to hold the intrinsic matrix and distortion coefficients
    intrinsics_matrix = None
    dist_coeffs = None

    # Parse the lines to find the intrinsics and distortion coefficients
    for i, line in enumerate(lines):
        if "Intrinsics are:" in line:
            matrix_lines = lines[i+1:i+4]
            matrix = []
            for mat_line in matrix_lines:
                clean_line = mat_line.replace('[', '').replace(']', '').strip()
                matrix.extend([float(num) for num in clean_line.split()])
            intrinsics_matrix = np.array(matrix).reshape(3, 3)
        
        if "Distortion matrix is:" in line:
            dist_line = lines[i+1].replace('[', '').replace(']', '').strip()
            dist_coeffs = np.array([float(num) for num in dist_line.split()])

    # Ensure that both intrinsics and distortion were found
    if intrinsics_matrix is None or dist_coeffs is None:
        raise ValueError("Failed to find both intrinsics and distortion matrix in the file.")
    
    return intrinsics_matrix, dist_coeffs

intrinsics_path2 = 'results/video1308/Filtered_frames/intrinsics.txt'
camera_matrix, dist_coeffs = load_intrinsics(intrinsics_path2)

dist_coeffs = np.array([-0.59344657, 0.99158244, 0.00599652, 0.00992879, -1.53399657])  # Replace with actual distortion coefficients

# Step 2: Detect Charuco corners in an image
def detect_and_visualize_charuco(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image {image_path}")
        return None, None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    
    if len(corners) > 0:
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)
            print(f"Detected {len(charuco_corners)} Charuco corners.")
            cv2.imshow('Detected Charuco Corners', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return charuco_corners, charuco_ids
        else:
            print("Charuco corners were not detected.")
            return None, None
    else:
        print("ArUco markers were not detected.")
        return None, None

# Step 3: Process all images in a folder
def process_images(image_files):
    for image_file in image_files:
        charuco_corners, charuco_ids = detect_and_visualize_charuco(image_file)
        if charuco_corners is None:
            print(f"Failed to detect Charuco corners in {image_file}")

# Provide the path to the folder containing your images
image_files = glob.glob('results/video1308/Filtered_frames/*.png')  # Adjust the path as necessary

process_images(image_files)
