"""
Date: June 17th - July 23rd 2024
Author: Rudrapriya Padmanabhan

Hand eye matrix calibration, using cv2 functions
Obtain the hand eye matrix for reprojection and camera trajectory
"""

import numpy as np
import pandas as pd
import cv2

# Step 1: Load the data
em_data_path = 'results/video1308/acquisition/intriniscs_refinement/matched_sensor_data.csv'
extrinsics_data_path = 'results/video1308/acquisition/intriniscs_refinement/extrinsics.csv'

# Load CSV files
em_data = pd.read_csv(em_data_path)
extrinsics_data = pd.read_csv(extrinsics_data_path)

# Step 2: Align the EM data with the extrinsics data
# merge the extrinsics data with the EM data on the frame number
aligned_data = pd.merge(extrinsics_data, em_data, left_on='Frame', right_on='Count')

# Step 3: Parse the transformation matrices from the aligned data
def parse_matrices_from_aligned_data(aligned_df):
    A_matrices = []
    B_matrices = []
    
    for _, row in aligned_df.iterrows():
        # Parse A matrix (from EM data)
        A_matrix = np.eye(4)
        A_matrix[:3, :3] = np.array([[row['a11_y'], row['a12_y'], row['a13_y']],
                                     [row['a21_y'], row['a22_y'], row['a23_y']],
                                     [row['a31_y'], row['a32_y'], row['a33_y']]])
        A_matrix[:3, 3] = np.array([row['x'], row['y'], row['z']])
        A_matrices.append(A_matrix)
        
        # Parse B matrix (from extrinsics)
        B_matrix = np.array([[row['a11_x'], row['a12_x'], row['a13_x'], row['a14']],
                             [row['a21_x'], row['a22_x'], row['a23_x'], row['a24']],
                             [row['a31_x'], row['a32_x'], row['a33_x'], row['a34']],
                             [row['a41'], row['a42'], row['a43'], row['a44']]])
        B_matrices.append(B_matrix)
    
    return A_matrices, B_matrices

# Parse the matrices
A_matrices, B_matrices = parse_matrices_from_aligned_data(aligned_data)

# Step 4: Perform hand-eye calibration
def hand_eye_calibration(A_matrices, B_matrices):
    # Convert lists to numpy arrays for OpenCV function
    A_rotations = np.array([A[:3, :3] for A in A_matrices])
    A_translations = np.array([A[:3, 3] for A in A_matrices])
    B_rotations = np.array([B[:3, :3] for B in B_matrices])
    B_translations = np.array([B[:3, 3] for B in B_matrices])

    # Perform hand-eye calibration using the Daniilidis method
    R_hand_eye, t_hand_eye = cv2.calibrateHandEye(
        R_gripper2base=A_rotations, t_gripper2base=A_translations,
        R_target2cam=B_rotations, t_target2cam=B_translations,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS
    )
    
    # Construct the hand-eye transformation matrix
    hand_eye_matrix = np.eye(4)
    hand_eye_matrix[:3, :3] = R_hand_eye
    hand_eye_matrix[:3, 3] = t_hand_eye.flatten()

    return hand_eye_matrix, R_hand_eye, t_hand_eye

# Perform the hand-eye calibration
hand_eye_matrix, R_hand_eye, t_hand_eye = hand_eye_calibration(A_matrices, B_matrices)

# Output the results
print("Hand-Eye Calibration Matrix:\n", hand_eye_matrix)
print("Rotation:\n", R_hand_eye)
print("Translation:\n", t_hand_eye)

