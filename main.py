"""
Date: June 5th 2024
Author: Rudrapriya Padmanabhan
To note: Logic for loading NDI data was provided by Laurent Mennillo 

"""
import cv2 as cv
from sksurgerynditracker.nditracker import NDITracker
# from sksurgerytrackervisualisation.ui import sksurgerytrackervisualisation_cl
import keyboard
import pathlib
import sys
import numpy as np
import pandas as pd
import time
import datetime

def main():
    out_path = pathlib.Path("./acquisition")

    acquisition_path = out_path / "CBH"

    date_path = acquisition_path / datetime.date.today().__str__()
    time_path = date_path / datetime.datetime.now().strftime("%H%M").__str__()

    video_path = time_path / "video"
    em_path = time_path / "EM"


    if not out_path.exists():
        out_path.mkdir()
    
    if not acquisition_path.exists():
        acquisition_path.mkdir()
    
    if not date_path.exists():
        date_path.mkdir()

    if not time_path.exists():
        time_path.mkdir()

    if not video_path.exists():
        video_path.mkdir()

    if not em_path.exists():
        em_path.mkdir()

    # ----- Open VideoCapture object -----

    # Select camera ID
    video_capture = cv.VideoCapture(0, cv.CAP_DSHOW)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH,1920)
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT,1080) 

    # We need to check if camera is already open
    if (video_capture.isOpened() == False): 
        print("Error reading video file") 

    # Set video resolutions
    frame_width = int(video_capture.get(3)) #1920
    frame_height = int(video_capture.get(4)) #1080
    size = (1920, 1080) 

    result = cv.VideoWriter(str(video_path/'video.avi'), 
                            cv.VideoWriter_fourcc(*'X264'), 
                            30, size) 

    # ----- Open NDITracker object -----

    settings = {
        "tracker type": "aurora",
        "ports to probe": 2,
        "verbose": True,
        "use quaternions": False
    }
    tracker = NDITracker(settings)
    tracker.start_tracking()


    # ----- Start capture and tracking -----

    count = 0
    data_list = []
    times_list = []
    while True:
        start = time.time()
        ret, frame = video_capture.read()

        if not ret:
            print("In capture_frame(): cv.VideoCapture.read() failed.")
            return cv.Mat

        port_handles, timestamps, framenumbers, tracking, quality = tracker.get_frame()

        for t in tracking:
            print(t)

        if ret is True:

            # Write the frame into the file 
            result.write(frame) 

            # Display the frame q
            # saved in the file 
            #cv.imshow('Frame', frame) 
            #cv.waitKey(1)
        
        for track in tracking:
            # Extract a11 to a33 and x, y, z for each tracker
            a_values = track[:3, :3].flatten()
            xyz_values = track[:3, 3]
            
            # Combine a_values and xyz_values into one list
            combined_values = np.concatenate((a_values, xyz_values))
            
            # Append the timestamp (or count) and combined values to the data list
            data_list.append([timestamps[0]]+[count] + combined_values.tolist())

        

        count += 1
        end = time.time()
        fps = 1/(end-start)
        times_list.append([fps])
    

        if keyboard.is_pressed('q'):
            break
    columns = ['Timestamp'] + ['Count'] + [f'a{i}{j}' for i in range(1, 4) for j in range(1, 4)] + ['x', 'y', 'z']
    df = pd.DataFrame(data_list, columns=columns)

    # Save the DataFrame to a CSV file
    df.to_csv(em_path / 'em_data.csv', index=False)
    df_times = pd.DataFrame(times_list,columns = ['times'])
    df_times.to_csv(em_path/'fps.csv', index=False)
    # ----- Close VideoCapture object -----

    video_capture.release()
    cv.destroyAllWindows()
    result.release() 

    # ----- Close NDITracker object -----

    tracker.stop_tracking()
    tracker.close()

    sys.exit(0)


if __name__ == '__main__':
    main()

