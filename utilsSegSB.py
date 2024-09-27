#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:53:19 2020

@author: sophiabano
"""
from torch.utils.data import Dataset as BaseDataset
from matplotlib import pyplot as plt
import albumentations as albu
import numpy as np
import os
import cv2

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
##############################################################################    
# data augmentation
# Since our dataset is very small we will apply a large number of different augmentations:
# horizontal flip
# affine transforms
# perspective transforms
# brightness/contrast/colors manipulations
# image bluring and sharpening
# gaussian noise
# random crops
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=0.7, border_mode=0),

        albu.Affine(shear=0.05, p=0.7),
        albu.PadIfNeeded(min_height=448, min_width=448, always_apply=True, border_mode=0, value=0),
        albu.RandomCrop(height=224, width=224, always_apply=True),
        
        albu.RandomBrightnessContrast (brightness_limit=0.1, contrast_limit=0.1, p=0.7),
        
        #albu.IAAAdditiveGaussianNoise(p=0.2),



    ]

    return albu.Compose(train_transform)

                    
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(448, 448)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    #class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, prec, rec, f1, iou

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou


##
##


class ConfusionMatrix:
    """
    Class that calculates the confusion matrix.
    It keeps track of computed confusion matrix
    until it has been reseted. 
    The ignore label should always be >= num_classes
    :param num_classes: [int] Number of classes
    :param: confusion_matrix: 2D ndarray of confusion_matrix
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def get_confusion_matrix(self):
        """
        Returns confusion matrix
        :return: confusion_matrix: 2D ndarray of confusion_matrix
        """
        return self.confusion_matrix

    def update_confusion_matrix(self, gt_mask, pre_mask):
        """
        Calculates the confusion matrix for a given ground truth
        and predicted segmentation mask and updates it
        :param gt_mask: 2D ndarray of ground truth segmentation mask
        :param pre_mask: 2D ndarray of predicted segmentation mask
        :return: confusion_matrix: 2D ndarray of confusion_matrix
        """
        assert gt_mask.shape == pre_mask.shape, f" {gt_mask.shape} == {pre_mask.shape}"

        mask = (gt_mask >= 0) & (gt_mask < self.num_classes)
        label = self.num_classes * gt_mask[mask].astype("int") + pre_mask[mask].astype("int")
        count = np.bincount(label, minlength=self.num_classes ** 2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)
        return self.confusion_matrix


def pixel_accuracy(confusion_matrix):
    """
    Calculates mean intersection over union given
    the confusion matrix of ground truth and predicted
    segmentation masks
    :param confusion_matrix: 2D ndarray of confusion_matrix
    :return: acc: [float] pixel accuracy
    """
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return acc


def pixel_accuracy_class(confusion_matrix):
    """
    Calculates pixel accuracy per class given
    the confusion matrix of ground truth and predicted
    segmentation masks
    :param confusion_matrix: 2D ndarray of confusion_matrix
    :return: acc: [float] mean pixel accuracy per class
    """
    acc = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1)+1e-10)
    acc = np.nanmean(acc)
    return acc


def mean_intersection_over_union(confusion_matrix):
    """
    Calculates mean intersection over union given
    the confusion matrix of ground truth and predicted
    segmentation masks
    :param confusion_matrix: 2D ndarray of confusion_matrix
    :return: miou: [float] mean intersection over union
    """
    miou = np.diag(confusion_matrix) / ((np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)) + 1e-10)
    miou = np.nanmean(miou)
    return miou


def segmentation_metrics(gt_masks, pred_masks, num_classes):
    """
    Calculates segmentation metrics (pixel accuracy, pixel accuracy per class,
    and mean intersection over union) for a list of ground truth and predicted
    segmentation masks for a given number of classes
    :param gt_masks: [list] 2D ndarray of ground truth segmentation masks
    :param pred_masks: [list] 2D ndarray of predicted segmentation masks
    :param num_classes: [int] Number of classes
    :return: pa, pac, miou [float, float, float]: metrics
    """
    assert len(gt_masks) == len(pred_masks)
    confusion_matrix = ConfusionMatrix(num_classes=num_classes)

    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        confusion_matrix.update_confusion_matrix(gt_mask, pred_mask)

    cm = confusion_matrix.get_confusion_matrix()
    pa = pixel_accuracy(cm)
    pac = pixel_accuracy_class(cm)
    miou = mean_intersection_over_union(cm)
    return pa, pac, miou
