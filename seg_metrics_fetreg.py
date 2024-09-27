"""
Segmentation metrics calculation script
for FetReg 2021
"""
import numpy as np
import cv2
import os

from visualisation_fetreg2021 import plot_image_n_label, plot_image_gt_pred_labels



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
    acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
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
    miou = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
        )
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


def segmentation_metrics_fetreg(gt_masks, pred_masks):
    """
    Calculates segmentation metrics (pixel accuracy, pixel accuracy per class,
    and mean intersection over union) for a list of ground truth and predicted
    segmentation masks 
    :param gt_masks: [list] 2D ndarray of ground truth segmentation masks
    :param pred_masks: [list] 2D ndarray of predicted segmentation masks
    :return: pa, pac and miou: [float, float, float] metrics
    """
    pa, pac, miou = segmentation_metrics(gt_masks=gt_masks,
                                         pred_masks=pred_masks,
                                         num_classes=4)
    return pa, pac, miou


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--Input_path", help="Path to ground truth label folder", default = "../dataset/data/Video023/images")#required=True)
    parser.add_argument("--GT_path", help="Path to ground truth label folder", default = "../dataset/data/Video023/labels")#required=True)
    parser.add_argument("--output", help="Path to folder with predicted masks", default = "../dataset/data/Video023/predicted_mask")# required=True)
    parser.add_argument("--vis_path", help="Output path to save plot", default = "../vis")# required=True)

    args = parser.parse_args()
   
    Input_path = args.Input_path
    GT_path = args.GT_path
    Pred_path = args.output
    Vis_path = args.vis_path
    
    if not os.path.exists(Vis_path):
        os.makedirs(Vis_path)
        print(Vis_path +' created')
    else:
        print(Vis_path +' exists')
        
    ids = os.listdir(Pred_path)
    
    # Reading ground truth and predicted masks and calculating metrics for FetReg
    gt = []
    prediction = []
    for i in range(len(ids)):
        image_vis = cv2.imread(Input_path+'/' +ids[i])
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(GT_path + '/' + ids[i], 0)
        gt.append(gt_mask)
        
        pred_mask = cv2.imread(Pred_path + '/' + ids[i], 0)
        prediction.append(pred_mask)
        
        #fig = plot_image_gt_pred_labels(image_vis, gt_mask, pred_mask)
        #fig.savefig(os.path.join(Vis_path+'/' +ids[i]))
        
    metrics = segmentation_metrics_fetreg(gt, prediction)
    print("PA = ", metrics[0])
    print("PAC = ", metrics[1])
    print("MIOU = ", metrics[2])

