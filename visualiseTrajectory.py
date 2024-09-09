'''
Date: September 1st 2024
Author: Rudrapriya Padmanabhan
Visualising the camera and the mosaic trajecotry on an XY plane 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the updated camera trajectory and affine transformation CSV files
camera_trajectory = pd.read_csv('100/camera_trajectory_affine.csv')
affine_transformations = pd.read_csv('100/frame_transformations.csv')

# Function to extract 2D translation from a 3x3 matrix
def extract_translation_from_affine(df):
    x = df['M13'].values
    y = df['M23'].values
    return x, y

# Extract ground truth and estimated translation vectors
x_gt, y_gt = extract_translation_from_affine(camera_trajectory)
x_est, y_est = extract_translation_from_affine(affine_transformations)

# Plot the trajectories
plt.figure(figsize=(10, 6))
plt.plot(x_gt, y_gt, label='Ground Truth Trajectory (XY Plane)', color='blue', marker='o')
plt.plot(x_est, y_est, label='Estimated Trajectory (XY Plane)', color='red', marker='x')

# Add labels and title
plt.xlabel('X Position (XY Plane)')
plt.ylabel('Y Position (XY Plane)')
plt.title('Ground Truth vs Estimated Trajectory')
plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig('100/trajectory_comparison.png')

# Show the plot
plt.show()

print("Trajectory comparison plot saved as 'trajectory_comparison.png'")
