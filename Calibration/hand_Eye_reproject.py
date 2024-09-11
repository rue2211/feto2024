"""
Date: June 17th - August 23rd 2024
Author: Rudrapriya Padmanabhan
Reprojection error visualisation, attempt to project the results of the hand eye matrix onto a charuco board
We are refining the quality of the reprojection via least squares 
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
import pathlib


camera_matrix = np.array([[697.60984021, 0., 908.75490056],
                          [0., 696.4033242, 491.26091482],
                          [0., 0., 1.]])
dist_coeffs = np.array([-0.54288582,  0.5190286,  0.00870424,  0.01297885, -0.44121777])

hand_eye_matrix = np.array( [[ 9.88355635e-01, -5.02167980e-03,  1.52078668e-01,  7.04768353e+01],
 [-1.33485529e-01,  4.51136106e-01,  8.82415904e-01,  3.95961647e+02],
 [-7.30393883e-02, -8.92441032e-01,  4.45212591e-01,  1.65310502e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# Define the Charuco board
charuco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
charuco_board = cv2.aruco.CharucoBoard_create(9, 7, 5, 3, charuco_dict)

image_folder = pathlib.Path("results/video1308/acquisition/intriniscs_refinement")


detected_image_points = []
detected_object_points = []

def process_frame(framepath):
    frame = cv2.imread(str(framepath))
    if frame is None:
        print(f"Frame {framepath} not found")
        return None, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, charuco_dict)

    if ids is not None and len(ids) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
        if charuco_corners is not None and len(charuco_corners) > 0:
            # Initialize rvec and tvec as empty arrays
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
            
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs, rvec, tvec
            )
            if retval:
                object_points = charuco_board.chessboardCorners[charuco_ids.flatten()]
                return frame, charuco_corners, object_points, rvec, tvec
    return None, None, None, None, None

#apply hand eye to points so we can see the transformation and size of error
def apply_hand_eye_matrix(rvec, tvec, hand_eye_matrix):
    # Convert rvec to rotation matrix
    R_cam, _ = cv2.Rodrigues(rvec)
    T_cam = np.hstack((R_cam, tvec))  # 3x4 matrix

    # Convert to 4x4 matrix for transformation
    T_cam = np.vstack((T_cam, [0, 0, 0, 1]))

    # Apply hand-eye transformation
    T_hand = hand_eye_matrix @ T_cam

    # Extract rotation and translation from the resulting matrix
    R_hand = T_hand[:3, :3]
    t_hand = T_hand[:3, 3].reshape(3, 1)

    # Convert back to rvec
    rvec_hand, _ = cv2.Rodrigues(R_hand)

    return rvec_hand, t_hand

def reprojection_error(params, object_points, image_points, camera_matrix, dist_coeffs):
    rvec, tvec = params[:3], params[3:]
    reprojected_image_points, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    reprojected_image_points = reprojected_image_points.reshape(-1, 2)
    error = image_points - reprojected_image_points
    return error.flatten()

def main():
    image_files = sorted(image_folder.glob("*.png"))
    
    total_error = 0.0
    total_points = 0
    
    for image_file in image_files:
        frame, image_points, object_points, rvec, tvec = process_frame(image_file)
        if frame is not None:
            detected_image_points.append(image_points)
            detected_object_points.append(object_points)

            # Apply hand-eye matrix to rvec and tvec
            rvec_hand, tvec_hand = apply_hand_eye_matrix(rvec, tvec, hand_eye_matrix)

            # Refine pose using solvePnP with iterative method
            _, rvec_refined, tvec_refined = cv2.solvePnP(
                object_points, image_points, camera_matrix, dist_coeffs,
                rvec_hand, tvec_hand, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
            )

            # Reproject the 3D points to 2D using the refined pose
            reprojected_image_points, _ = cv2.projectPoints(
                object_points, rvec_refined, tvec_refined,
                camera_matrix, dist_coeffs
            )
            reprojected_image_points = reprojected_image_points.reshape(-1, 2)

            # Visualize reprojection on the current frame
            for j in range(len(image_points)):
                detected_point = tuple(image_points[j].ravel().astype(int))
                reprojected_point = tuple(reprojected_image_points[j].ravel().astype(int))

                print(f"Detected point: {detected_point}, Reprojected point: {reprojected_point}")

                cv2.circle(frame, detected_point, 5, (0, 255, 0), -1)
                cv2.circle(frame, reprojected_point, 5, (0, 0, 255), -1)
                cv2.line(frame, detected_point, reprojected_point, (255, 0, 0), 2)

            error = np.linalg.norm(image_points - reprojected_image_points, axis=1)
            total_error += np.sum(error)
            total_points += len(error)

            cv2.imshow(f'Reprojected Frame', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    mean_error = total_error / total_points if total_points > 0 else 0
    print(f"Mean reprojection error: {mean_error}")

if __name__ == '__main__':
    main()
