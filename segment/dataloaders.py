#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:10:36 2020

@author: sophiabano
"""
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import os
import cv2 
import matplotlib.pyplot as plt

##############################################################################  

# RGB to gray mask
def rgb2mask(img,color2index):

    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3

    W = np.power(256, [[0],[1],[2]])

    img_id = img.dot(W).squeeze(-1) 
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)

    for i, c in enumerate(values):
        try:
            mask[img_id==c] = color2index[tuple(img[img_id==c][0])] 
        except:
            pass
    return mask

# RGB to gray mask
def mask2rgb(mask, index2color):

    if len(mask.shape) == 3:
        mask =np.max(mask,axis=0)
    height, width = mask.shape
    values = np.unique(mask)
    
    img_mask = np.zeros([3, height, width])
    for i in range(height):
        for j in range(width):
            img_mask[:, i, j] = index2color[mask[i, j]]
            
    return img_mask


def color_convCV():  
    #Color coding for Camvid dataset
    color2index = {
        (255, 0, 0) : 1, #'vessel', 
        (0,   0,  0) : 0, #'background/unlabel'
    }
    
    index2color = {
        1 : (255, 0, 0), #'vessel', 
        0: (0,   0,   0), #'background'

    }
    return color2index, index2color


##############################################################################    
# Data loader 
class DatasetCV_test(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['background', 'vessel', 'tool','fetus']
    
    def __init__(
            self, 
            images_dir, 
            #masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        #self.ids = os.listdir(images_dir)
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        if isinstance(images_dir, list):
            self.images_fps = images_dir  # Directly use the list of file paths
            self.ids = [os.path.basename(path) for path in images_dir]  # Extract file names
        else:
            self.ids = os.listdir(images_dir)
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Smoothing for Video013
        #kernel = np.ones((5,5),np.float32)/25
        #image = cv2.filter2D(image,-1,kernel)
        
        image = cv2.resize(image, (448, 448), interpolation = cv2.INTER_LINEAR)

        ids = self.ids[i]
        
        #mask = cv2.imread(self.masks_fps[i],0)
        #mask = image.copy()
        mask = np.random.randint(1, size=(image.shape[0], image.shape[1]), dtype=np.uint8)
        

        #print(np.unique(mask[:,:,0]))
        #print(np.unique(mask[:,:,1]))
        #print(np.unique(mask[:,:,2]))

        #color2index, index2color = color_convCV()
        #mask = rgb2mask(mask,color2index)
        # extract certain classes from mask (e.g. cars)

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask, ids
        
    def __len__(self):
        #return len(self.ids)
        return len(self.images_fps)
##############################################################################    
# Data loader 
class DatasetCV(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['background', 'vessel', 'tool','fetus']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
            # Check if directories or lists are provided
        if isinstance(images_dir, str):
            self.ids = os.listdir(images_dir)
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        else:
            self.images_fps = images_dir
            self.ids = [os.path.basename(path) for path in self.images_fps]

        if isinstance(masks_dir, str):
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        else:
            self.masks_fps = masks_dir
    
        #self.ids = os.listdir(images_dir)
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        if len(self.images_fps) != len(self.masks_fps):
            raise ValueError("Mismatch between number of images and masks")

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):

        if i >= len(self.masks_fps):
            raise IndexError(f"Index {i} is out of range for dataset length {len(self.masks_fps)}")
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (448, 448), interpolation = cv2.INTER_LINEAR)
        
        #mask_str = self.masks_fps[i]
        # mask_str = mask_str.replace('.png','_mask.png')
       
        mask = cv2.imread(self.masks_fps[i],0)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask= cv2.resize(mask, (448, 448), interpolation = cv2.INTER_NEAREST)
        mask = np.clip(mask, 0, 1).astype(np.float32)

        #print(np.unique(mask[:,:,0]))
        #print(np.unique(mask[:,:,1]))
        #print(np.unique(mask[:,:,2]))

        # color2index, index2color = color_convCV()
        # mask = rgb2mask(mask,color2index)
        # extract certain classes from mask (e.g. vessel)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        #if len(self.class_values) == 1:
        #    mask_inv = 1- mask
        #    mask = np.concatenate((mask_inv,mask), axis = 2)
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        #return len(self.ids)
        return len(self.images_fps)
