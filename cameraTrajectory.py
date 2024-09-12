
"""
Date: August 31st 2024
Author: Rudrapriya Padmanabhan

Script to calculate the camera trajectory
Uses the hand eye data and the EM data to do matrix multiplication 
"""

import numpy as np
import pandas as pd

# Load the sensor data
sensor_data = pd.read_csv('500/EMdata.csv')

# remove NaN
sensor_data_clean = sensor_data.dropna()

# Hand-eye calibration matrix 
hand_eye_matrix = np.array([[ 9.88355635e-01, -5.02167980e-03,  1.52078668e-01,  7.04768353e+01],
                            [-1.33485529e-01,  4.51136106e-01,  8.82415904e-01,  3.95961647e+02],
                            [-7.30393883e-02, -8.92441032e-01,  4.45212591e-01,  1.65310502e+02],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# take sensor data and make into 4x4
def create_transformation_matrix(rotation, translation):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    return transformation_matrix

# get camera pose
def compute_camera_pose(sensor_rotation, sensor_translation, hand_eye_matrix):
    sensor_transformation = create_transformation_matrix(sensor_rotation, sensor_translation)
    camera_pose = np.dot(sensor_transformation, hand_eye_matrix)
    return camera_pose

# store trajectory in a list
camera_trajectory_3d = []

# Iterate over each row of the clean sensor data
for _, row in sensor_data_clean.iterrows():
    # Extract the rotation matrix
    sensor_rotation = np.array([
        [row['a11'], row['a12'], row['a13']],
        [row['a21'], row['a22'], row['a23']],
        [row['a31'], row['a32'], row['a33']]
    ])
    
    # Extract the translation vector
    sensor_translation = np.array([row['x'], row['y'], row['z']])
    
    # Calculate the camera pose
    camera_pose = compute_camera_pose(sensor_rotation, sensor_translation, hand_eye_matrix)
    
    # Extract rotation (3x3) and translation (1x3) from the camera pose
    rotation_part = camera_pose[:3, :3].flatten()
    translation_part = camera_pose[:3, 3]
    
    # combine rotation and translation into a single list
    pose = np.concatenate([rotation_part, translation_part])
    
    # Append trajcetory list 
    camera_trajectory_3d.append(pose)

# names of rotation and translation 
column_names = [
    "R11", "R12", "R13",
    "R21", "R22", "R23",
    "R31", "R32", "R33",
    "Tx", "Ty", "Tz"
]

# list -> DF
trajectory_df = pd.DataFrame(camera_trajectory_3d, columns=column_names)

# save as CSV
trajectory_df.to_csv('500/camera_trajectory_3d.csv', index=False)

print("Camera trajectory (rotation and translation) has been exported to 'camera_trajectory_3d.csv'")
