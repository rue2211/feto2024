"""
Date:September 1st 2024
Author: Rudrapriya Padmanabhan
Align the trajectories, find their complete orientation 
We then calculate the ATE,RTE and rotaiton error for the best homography decompositon and the camera trajectory for a sequence
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_csv_data(trajectory_csv, solution_csv):
    """Load the trajectory and decomposed solution CSV files."""
    camera_trajectory_df = pd.read_csv(trajectory_csv)
    decomposed_solution_df = pd.read_csv(solution_csv)
    return camera_trajectory_df, decomposed_solution_df

def extract_3d_points(df, is_trajectory=True):
    """
    Extract 3D points from the dataframe.
    If it's a camera trajectory CSV, we use 'Tx', 'Ty', 'Tz'.
    If it's a decomposed solution CSV, we use 'T1', 'T2', 'T3'.
    """
    if is_trajectory:
        return df[['Tx', 'Ty', 'Tz']].to_numpy()
    else:
        return df[['T1', 'T2', 'T3']].to_numpy()

def compute_orientation(x1, x2):
    """
    Compute absolute orientation between two sets of 3D points.
    """
    x1_mean = np.mean(x1, axis=0)
    x2_mean = np.mean(x2, axis=0)

    H = np.dot((x2 - x2_mean).T, (x1 - x1_mean))
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = x1_mean - np.dot(R, x2_mean)
    s = np.sum(S) / np.trace(np.dot(x2.T, x2))  # Fixed scale factor calculation

    return R, t, s

def translate_to_origin(points):
    """Translate a trajectory so that it starts at the origin."""
    return points - points[0]

def compute_ate(estimated_trajectory, ground_truth_trajectory):
    """Compute Absolute Trajectory Error (ATE)."""
    errors = np.linalg.norm(estimated_trajectory - ground_truth_trajectory, axis=1)
    return np.mean(errors)

def compute_rte(estimated_trajectory, ground_truth_trajectory):
    """Compute Relative Pose Error (RTE)."""
    n = estimated_trajectory.shape[0]
    relative_errors = []
    
    for i in range(n - 1):
        delta_est = np.linalg.norm(estimated_trajectory[i+1] - estimated_trajectory[i])
        delta_gt = np.linalg.norm(ground_truth_trajectory[i+1] - ground_truth_trajectory[i])
        relative_errors.append(abs(delta_est - delta_gt))
    
    return np.mean(relative_errors)

def rotation_matrix_to_angle(R):
    """Convert a rotation matrix to an angle (in radians) representing the magnitude of rotation."""
    angle = np.arccos((np.trace(R) - 1) / 2)
    return angle

def compute_rot(estimated_trajectory, ground_truth_trajectory, R):
    """Compute Rotation Error (ROT) between the predicted and ground truth poses."""
    n = estimated_trajectory.shape[0]
    rotation_errors = []
    
    for i in range(n):
        # Here we assume the ground truth poses provide rotation matrices
        R_gt = np.eye(3)  # This should be replaced with the actual ground truth rotation at index i
        
        # The relative rotation matrix
        R_rel = np.dot(R_gt.T, R)  # R_gt^T * R gives the relative rotation
        angle = rotation_matrix_to_angle(R_rel)
        
        rotation_errors.append(angle)
    
    return np.mean(rotation_errors)



def plot_ate_rte_over_time(estimated_trajectory, ground_truth_trajectory):
    """Plot ATE and RTE over time."""
    n = estimated_trajectory.shape[0]
    ate_over_time = np.linalg.norm(estimated_trajectory - ground_truth_trajectory, axis=1)
    rte_over_time = [0]  # No relative error at the first point
    for i in range(1, n):
        delta_est = np.linalg.norm(estimated_trajectory[i] - estimated_trajectory[i-1])
        delta_gt = np.linalg.norm(ground_truth_trajectory[i] - ground_truth_trajectory[i-1])
        rte_over_time.append(abs(delta_est - delta_gt))
    
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(ate_over_time, label='ATE over time')
    plt.ylabel('ATE')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(rte_over_time, label='RTE over time', color='orange')
    plt.ylabel('RTE')
    plt.legend()
    
    plt.xlabel('Frame')
    plt.show()

def apply_transformation(mosaic_points, R, t, s):
    """Apply the computed transformation (rotation, translation, scale) to the mosaic points."""
    transformed_points = s * np.dot(R, mosaic_points.T).T + t.T
    return transformed_points

def plot_trajectories(camera_points, mosaic_points, transformed_mosaic_points):
    """Plot the camera and mosaic trajectories in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original camera trajectory (ground truth)
    ax.plot(camera_points[:, 0], camera_points[:, 1], camera_points[:, 2], label='Camera Trajectory', color='b')

    # Plot original mosaic trajectory (before alignment)
    ax.plot(mosaic_points[:, 0], mosaic_points[:, 1], mosaic_points[:, 2], label='Mosaic Trajectory (Before)', color='r')

    # Plot transformed mosaic trajectory (after alignment)
    ax.plot(transformed_mosaic_points[:, 0], transformed_mosaic_points[:, 1], transformed_mosaic_points[:, 2], 
            label='Mosaic Trajectory (After)', color='g')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()
    plt.title('3D Trajectory Comparison: Camera vs Mosaic')
    plt.show()

def plot_2d_trajectories(camera_points, mosaic_points, transformed_mosaic_points):
    """Plot the camera and mosaic trajectories in 2D (xy-plane, xz-plane, yz-plane)."""
    
    # Plot in xy-plane
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(camera_points[:, 0], camera_points[:, 1], label='Camera Trajectory', color='b')
    plt.plot(mosaic_points[:, 0], mosaic_points[:, 1], label='Mosaic Trajectory (Before)', color='r')
    plt.plot(transformed_mosaic_points[:, 0], transformed_mosaic_points[:, 1], label='Mosaic Trajectory (After)', color='g')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('XY Plane')
    plt.legend()

    # Plot in xz-plane
    plt.subplot(1, 3, 2)
    plt.plot(camera_points[:, 0], camera_points[:, 2], label='Camera Trajectory', color='b')
    plt.plot(mosaic_points[:, 0], mosaic_points[:, 2], label='Mosaic Trajectory (Before)', color='r')
    plt.plot(transformed_mosaic_points[:, 0], transformed_mosaic_points[:, 2], label='Mosaic Trajectory (After)', color='g')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('XZ Plane')
    plt.legend()

    # Plot in yz-plane
    plt.subplot(1, 3, 3)
    plt.plot(camera_points[:, 1], camera_points[:, 2], label='Camera Trajectory', color='b')
    plt.plot(mosaic_points[:, 1], mosaic_points[:, 2], label='Mosaic Trajectory (Before)', color='r')
    plt.plot(transformed_mosaic_points[:, 1], transformed_mosaic_points[:, 2], label='Mosaic Trajectory (After)', color='g')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('YZ Plane')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Load CSV data
camera_trajectory_csv = '300/camera_trajectory_3d.csv'
decomposed_solution_csv = '300/decomposed_solutions/decomposed_solution_2.csv'  # Best solution

camera_trajectory_df, decomposed_solution_df = load_csv_data(camera_trajectory_csv, decomposed_solution_csv)

# Extract the 3D points
camera_trajectory_points = extract_3d_points(camera_trajectory_df, is_trajectory=True)
decomposed_solution_points = extract_3d_points(decomposed_solution_df, is_trajectory=False)

# Ensure both sets have the same number of points
num_points = min(len(camera_trajectory_points), len(decomposed_solution_points))
camera_trajectory_points = camera_trajectory_points[:num_points]
decomposed_solution_points = decomposed_solution_points[:num_points]

# Translate both to start at the origin
camera_trajectory_points_origin = translate_to_origin(camera_trajectory_points)
decomposed_solution_points_origin = translate_to_origin(decomposed_solution_points)

# Compute orientation between corresponding 3D points
R, t, s = compute_orientation(camera_trajectory_points_origin, decomposed_solution_points_origin)

# Apply the transformation to align the mosaic trajectory with the camera trajectory
estimated_trajectory = apply_transformation(decomposed_solution_points_origin, R, t, s)

# Compute ATE and RTE and ROT
ate = compute_ate(estimated_trajectory, camera_trajectory_points_origin)
rte = compute_rte(estimated_trajectory, camera_trajectory_points_origin)
rot = compute_rot(estimated_trajectory, camera_trajectory_points_origin, R)

print(f"Absolute Trajectory Error (ATE): {ate}")
print(f"Relative Pose Error (RTE): {rte}")
print(f"Rotation Error (ROT): {rot}")
# Plot ATE and RTE over time
plot_ate_rte_over_time(estimated_trajectory, camera_trajectory_points_origin)

# Plot the camera trajectory and the mosaic trajectories (before and after transformation)
plot_trajectories(camera_trajectory_points_origin, decomposed_solution_points_origin, estimated_trajectory)

# Now call the function with the aligned and original points for 2D plotting
plot_2d_trajectories(camera_trajectory_points_origin, decomposed_solution_points_origin, estimated_trajectory)
