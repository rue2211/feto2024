"""
Date: August 20th 2024
Author: Rudrapriya Padmanabhan
Mean IOU test for segmentation model to evaluate the results from segmentation tests of different datasets
"""

import os 
import torch
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU
from dataloaders.dataloaders import DatasetCV
from utilsSegSB import get_preprocessing
import cv2
from tqdm import tqdm

ARCH = 'Unet++'
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ALL_CLASSES = ['background', 'vessel', 'tool', 'fetus']
CLASSES = ['background', 'vessel', 'tool', 'fetus']
CHECKPOINT_DIR = './checkpoints'
VIDEO_DIR = '/cs/student/projects1/cgvi/2023/rpadmana/FetReg2021Seg_RudraEdit/FetReg2021_dataset/Train/Train_FetReg2021_Task1_Segmentation/video1133'

# Derived Variables
class_values = [ALL_CLASSES.index(cls.lower()) for cls in CLASSES]
CHECK_PATH = os.path.join(CHECKPOINT_DIR, f'best_{ARCH}_{ENCODER}_{ACTIVATION}_BCE_CLASSES{len(class_values)}.pth')
x_valid_dir = os.path.join(VIDEO_DIR, 'images')
y_valid_dir = os.path.join(VIDEO_DIR, 'labels')

# Load the model with safe options
best_model = torch.load(CHECK_PATH)
#best_model.to(DEVICE)
best_model.eval()

# Preprocessing function for the model's encoder
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Prepare the dataset
valid_dataset = DatasetCV(
    images_dir=x_valid_dir,
    masks_dir=y_valid_dir,
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

# Initialize the IoU metric
iou_metric = IoU(threshold=0.5)

# Store IoU scores
iou_scores = []

# Compute IoU for each image
with torch.no_grad():
    for i, (image, gt_mask) in enumerate(tqdm(valid_loader, desc="Calculating IoU")):
        try:
            image = image.to(DEVICE)
            gt_mask = gt_mask.to(DEVICE)

            # Predict the mask
            pr_mask = best_model(image)
            pr_mask = (pr_mask > 0.5).float()  # Threshold for binary segmentation

            # Calculate IoU
            iou = iou_metric(pr_mask, gt_mask)
            iou_scores.append(iou.item())

            print(f"Image {i+1}/{len(valid_loader)} - IoU: {iou.item():.4f}")
        except Exception as e:
            print(f"Error processing image {i+1}: {e}")
            continue

# Calculate and display the average IoU
avg_iou = np.mean(iou_scores)
print(f"\nAverage IoU score for validation set: {avg_iou:.4f}")

