"""
Date: July 24th 2024
Author: Rudrapriya Padmanabhan

Script to rename files for sending to segmentation model 
"""
import os
import re

def rename_files(directory):
    # Get the name of the immediate parent directory
    parent_dir = os.path.basename(directory)
    
    # Find the maximum frame number to determine padding
    max_frame = 0
    for filename in os.listdir(directory):
        if filename.startswith("frame"):
            match = re.search(r'frame(\d+)', filename)
            if match:
                max_frame = max(max_frame, int(match.group(1)))
    
    # Determine padding based on the maximum frame number
    padding = len(str(max_frame))
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.startswith("frame"):
            # Extract the frame number
            match = re.search(r'frame(\d+)', filename)
            if match:
                frame_number = match.group(1)
                
                # Pad the frame number with zeros to maintain a minimum of 5 digits
                padded_frame = frame_number.zfill(5)
                
                # Construct the new filename
                new_filename = f"{parent_dir}_frame{padded_frame}"
                
                # Get the file extension
                _, extension = os.path.splitext(filename)
                new_filename += extension
                
                # Rename the file
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")

# Usage
directory_path = "results/video1308/video1308"
rename_files(directory_path)
