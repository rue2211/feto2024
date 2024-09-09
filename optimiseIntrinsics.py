"""
Author: Rudrapriya Padmanabhan
Date created: 3.July.2024
Performs refined camera calibration to our endoscope
Employs a charuco grid 
"""

import cv2
import numpy as np
import sksurgeryimage.calibration.charuco_point_detector as cpd
import sksurgerycalibration.video.video_calibration_driver_mono as mc
import sksurgerycalibration.video.video_calibration_utils as surg_utils
import pathlib
import pandas as pd
import re

# Calibrating using charuco board
chessboard_corners = (9, 7)
min_points_to_detect = 15
square_size_mm = (5, 3)
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

detector = cpd.CharucoPointDetector(dictionary, chessboard_corners, square_size_mm)
calibrator = mc.MonoVideoCalibrationDriver(detector, min_points_to_detect)

# Store detected corners and object points
detected_image_points = []
detected_object_points = []

def process_frame(framepath):
    frame = cv2.imread(framepath)
    if frame is None:
        print(f"Frame {framepath} not found")
        return None, None, None

    ids, object_points, image_points = detector.get_points(frame)
    if ids is not None and ids.shape[0] >= min_points_to_detect:
        return frame, image_points, object_points
    else:
        print(f"Insufficient points detected in frame: {framepath}")
    return None, None, None

def sort_numerically(file_list):
    """Sort filenames containing numbers in a way that respects numerical order."""
    def extract_numbers(f):
        numbers = re.findall(r'\d+', f)
        return [int(num) for num in numbers]
    
    return sorted(file_list, key=extract_numbers)

def main(image_folder):
    print(f"Image Folder: {image_folder}")

    # Manually filtered images, sorted numerically
    filtered_images = sort_numerically([str(p) for p in pathlib.Path(image_folder).glob("*.png")])
    
    for framepath in filtered_images:
        frame, image_points, object_points = process_frame(framepath)
        if frame is not None:
            detected_image_points.append(image_points)
            detected_object_points.append(object_points)
            calibrator.grab_data(frame)

    if calibrator.get_number_of_views() >= 2:
        print(f"Total number of views: {calibrator.get_number_of_views()}")
        proj_err, params = calibrator.calibrate()
        print(f"Projection error: {proj_err}")
        print(f'Intrinsics are: \n  {params.camera_matrix}')
        print(f'Distortion matrix is:  \n {params.dist_coeffs}')

        # Save intrinsics
        with open(image_folder / "intrinsics.txt", "a") as intrinsics_file:
            intrinsics_file.write(f"Reprojection error: {proj_err} \n")
            intrinsics_file.write(f'Intrinsics are: \n  {params.camera_matrix}\n')
            intrinsics_file.write(f'Distortion matrix is:  \n {params.dist_coeffs}\n')

        # Save extrinsics
        extr_list = []
        for i in range(len(params.rvecs)):
            extrinsics = surg_utils.extrinsic_vecs_to_matrix(params.rvecs[i], params.tvecs[i])
            # Extract frame number
            frame_number = int(pathlib.Path(filtered_images[i]).stem.replace('frame', ''))
            extr_list.append([frame_number] + extrinsics.flatten().tolist())
        columns = ['Frame'] + [f'a{i}{j}' for i in range(1, 5) for j in range(1, 5)]
        df = pd.DataFrame(extr_list, columns=columns)
        df.to_csv(image_folder / 'extrinsics.csv', index=False)
        print(f'Extrinsics saved.')

        # Check lengths of lists to avoid IndexError
        if len(detected_object_points) != len(params.rvecs):
            print(f"Warning: Mismatch in lengths of detected object points and rotation vectors.")
            print(f"Length of detected_object_points: {len(detected_object_points)}")
            print(f"Length of params.rvecs: {len(params.rvecs)}")
            print("This might cause an IndexError.")

        # Reproject the object points and visualize the results
        for i, framepath in enumerate(filtered_images):
            if i >= len(detected_object_points) or i >= len(params.rvecs):
                print(f"Skipping frame {framepath} due to insufficient data.")
                continue

            frame = cv2.imread(framepath)
            if frame is None:
                print(f"Frame {framepath} not found or unable to read.")
                continue

            print(f"Processing frame: {framepath}")
            print(f"Detected Image Points (frame {i}): {detected_image_points[i]}")
            print(f"Detected Object Points (frame {i}): {detected_object_points[i]}")

            # Reproject the 3D object points back to the image plane
            reprojected_image_points, _ = cv2.projectPoints(detected_object_points[i], params.rvecs[i], params.tvecs[i],
                                                            params.camera_matrix, params.dist_coeffs)
            reprojected_image_points = reprojected_image_points.reshape(-1, 2)

            print(f"Reprojected Image Points (frame {i}): {reprojected_image_points}")

            # Draw detected corners and lines linking to reprojected points
            for j, point in enumerate(detected_image_points[i]):
                cv2.circle(frame, tuple(point.astype(int)), 5, (0, 255, 0), -1)  # Detected points in green
                cv2.line(frame, tuple(point.astype(int)), tuple(reprojected_image_points[j].astype(int)), (255, 0, 0), 2)

            # Draw reprojected points
            for point in reprojected_image_points:
                cv2.circle(frame, tuple(point.astype(int)), 5, (0, 0, 255), -1)  # Reprojected points in red

            # Save or display the image
            # output_path = image_folder / f"result_{pathlib.Path(framepath).stem}.png"
            # cv2.imwrite(str(output_path), frame)
            cv2.imshow(f'Result Frame {pathlib.Path(framepath).stem}', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print('Camera Calibration Complete!')

    else:
        print("Not enough views found.")

if __name__ == '__main__':
    main(pathlib.Path("results/video1308/acquisition/intriniscs_refinement")) #path to your improved frames
