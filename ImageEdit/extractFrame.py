"""
Author: Rudrapriya Padmanabhan
Date created: 8.July.2024
Version with no image cropping or scaling
"""

import cv2 
import os 

# File Paths
path = "results/video1308/acquisition"
vid_path = cv2.VideoCapture(os.path.join(path, "video/video.avi"))

try:
    # creating a folder named frame_data inside path
    frame_data_path = os.path.join(path, 'new_frame_data')
    if not os.path.exists(frame_data_path): 
        os.makedirs(frame_data_path) 

# if not created then raise error 
except OSError: 
    print('Error: Creating directory of frame data') 

# frame 
currentframe = 0

while True: 
    # reading from frame 
    ret, frame = vid_path.read() 

    if ret: 
     
        # file name creation - ensuring it can be sorted
        name = os.path.join(frame_data_path, f'frame{currentframe:05d}.png')
        print(f'Creating... {name}') 

        # Save frame with the highest quality settings
        cv2.imwrite(name, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # PNG format, 0 compression for maximum quality

        # go to next frame
        currentframe += 1
    else: 
        break

# Release all space and windows once done 
vid_path.release() 
cv2.destroyAllWindows() 

# Define the square ROI coordinates (adjusted to capture the full circle)


# """
# Author: Rudrapriya Padmanabhan
# Date created: 8.July.2024
# Version with cropping and scaling
# """

# import cv2 
# import os 

# # File Paths
# path = "results/video1308"
# vid_path = cv2.VideoCapture(os.path.join(path, "acquisition/video/video.avi"))

# try:
#     # creating a folder named frame_data inside the specified path
#     frame_data_path = os.path.join(path, 'frame_data')
#     if not os.path.exists(frame_data_path): 
#         os.makedirs(frame_data_path) 

# # if not created then raise error 
# except OSError: 
#     print('Error: Creating directory of frame data') 

# # frame 
# currentframe = 0

# # Define the square ROI coordinates (adjusted to capture the full circle)
# x, y, w, h = 600, 180, 720, 720  # Adjust these values as needed




# while True: 
#     # reading from frame 
#     ret, frame = vid_path.read() 

#     if ret: 
#         # Crop the frame to the square ROI
#         cropped_frame = frame[y:y+h, x:x+w]

#         # Resize the cropped frame to 448x448 using high-quality interpolation
#         resized_frame = cv2.resize(cropped_frame, (448, 448), interpolation=cv2.INTER_CUBIC)

#         # Generate the file name for the frame
#         name = os.path.join(frame_data_path, f'frame{currentframe:05d}.png')
#         print(f'Creating... {name}') 

#         # Save the resized frame with the highest quality settings
#         cv2.imwrite(name, resized_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # PNG format, 0 compression for maximum quality

#         # Increase counter to track the number of frames created
#         currentframe += 1
#     else: 
#         break

# # Release all space and windows once done 
# vid_path.release() 
# cv2.destroyAllWindows() 
