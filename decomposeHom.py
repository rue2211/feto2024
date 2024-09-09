"""
Date: August 31st 2024
Author: Rudrapriya Padmanabhan

Script to decompose homography from affine transformations using the intrinsic calibration
"""
import cv2
import numpy as np
import pandas as pd
import csv
import os

# Assuming intrinsic camera matrix (replace with your actual calibration values)
K = np.array([[697.60984021, 0, 908.75490056],
              [0, 696.4033242, 491.26091482],
              [0,  0,  1]])  # Replace fx, fy, cx, cy with your actual intrinsic parameters

def load_transformations_from_csv(csv_filename):
    """Load transformations from CSV and convert to homography matrices."""
    df = pd.read_csv(csv_filename)
    transformations = []
    
    for index, row in df.iterrows():
        # Convert the affine transformation (2x3) to a homography (3x3)
        transform = np.array([[row['M11'], row['M12'], row['M13']],
                              [row['M21'], row['M22'], row['M23']],
                              [0, 0, 1]])  # Assuming affine, we add [0, 0, 1] row for homography
        transformations.append(transform)
    
    return transformations

def decompose_homographies(transformations, K, output_dir):
    """Decompose homographies and save all images' poses for each solution in separate files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist

    num_images = len(transformations)
    decomposed_solutions = {}  # Store decompositions for each solution

    for i, homography in enumerate(transformations):
        # Decompose homography using the intrinsic matrix K
        num_solutions, Rs, Ts, Ns = cv2.decomposeHomographyMat(homography, K)
        
        # For each solution (j), accumulate decomposed poses for all images
        for j in range(num_solutions):
            if j not in decomposed_solutions:
                decomposed_solutions[j] = []

            R = Rs[j]
            T = Ts[j]
            N = Ns[j]
            
            # Store the pose data for this image and solution
            decomposed_solutions[j].append({
                "Image_Index": i,
                "Rotation": R,
                "Translation": T,
                "Normal": N
            })

    # Save each solution's decompositions into separate CSV files
    for solution_index, poses in decomposed_solutions.items():
        output_csv = os.path.join(output_dir, f'decomposed_solution_{solution_index}.csv')
        save_solution_to_csv(poses, output_csv)

def save_solution_to_csv(poses, csv_filename):
    """Save all images' decomposed camera poses for a particular solution to a CSV file."""
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["Image_Index", "R11", "R12", "R13", "R21", "R22", "R23", "R31", "R32", "R33",
                  "T1", "T2", "T3", "N1", "N2", "N3"]
        writer.writerow(header)
        
        for pose in poses:
            R = pose["Rotation"]
            T = pose["Translation"]
            N = pose["Normal"]
            row = [pose["Image_Index"]] + R.flatten().tolist() + T.flatten().tolist() + N.flatten().tolist()
            writer.writerow(row)

# Example usage
csv_filename = "500/image_transformations.csv"  # Path to your CSV file
output_dir = "500/decomposed_solutions/"  # Output directory for decomposed camera poses

# Load transformations from CSV
transformations = load_transformations_from_csv(csv_filename)

# Decompose homographies and save each solution in its own CSV file
decompose_homographies(transformations, K, output_dir)
