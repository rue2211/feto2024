#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:05:05 2020

@author: sophiabano
"""

import os, shutil, cv2
from distutils.dir_util import copy_tree
import torch
import numpy as np
import random
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader

from utilsSegSB import get_training_augmentation, visualize, evaluate_segmentation, segmentation_metrics
from visualisation_fetreg2021 import plot_image_n_label, plot_image_gt_pred_labels
from dataloaders.dataloaders import DatasetCV, color_convCV, mask2rgb, DatasetCV_test
from utilsSegSB import get_preprocessing, get_validation_augmentation

from sklearn.model_selection import train_test_split 
from segmentation_models_pytorch import utils

#DATA_DIR = '../../../python_code/Seg_Models/datasets/CamVid/'
#Below was original
# DATA_DIR = '../FetReg2021_dataset/'
# DATASET = 'feto_placenta'
ROOT_DIR = '/cs/student/projects1/cgvi/2023/rpadmana/FetReg2021Seg_RudraEdit/FetReg2021_dataset'
TRAIN_DIR = os.path.join(ROOT_DIR, "Train", 'Train_FetReg2021_Task1_Segmentation')

MODE = 'predict'   # train, predict test
EPOCHS = 1000
n_epochs_stop = 100

# https://github.com/qubvel/segmentation_models.pytorch
ARCH = 'Unet' # 'DeepLabV3' #'FPN'
ENCODER = 'resnet50' #'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'

ACTIVATION = 'sigmoid' # sigmoid or None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

#NEW: collect image and label paths from subdirectories
image_paths = []
label_paths = []

# PRINT params
print("MODE -->", MODE)
print("ARCHITECTURE -->", ARCH)
print("ENCODER -->", ENCODER)
print("EPOCHS -->", EPOCHS) 

#train_str = ['Video001','Video002','Video003','Video004', 'Video005','Video006','Video007','Video008','Video009','Video011',
             #'Video013', 'Video014', 'Video016', 'Video017', 'Video018','Video019', 'Video022', 'Video023']
ALL_CLASSES = ['background', 'vessel', 'tool','fetus']
#CLASSES =  ['vessel']#'background', 'vessel', 'tool','fetus']  
CLASSES =  ['background', 'vessel', 'tool','fetus']  
class_values = [ALL_CLASSES.index(cls.lower()) for cls in CLASSES]
#color2index, index2color = color_convCV()

#val_str = 'All'
# val_str_val = ['Video001','Video006','Video016'] # fold 1
# val_str_val = ['Video002','Video018','Video011']  # fold 2
# val_str_val = ['Video019','Video004','Video023']  # fold 3
#val_str_val = ['Video003','Video014','Video005']  # fold 4
# val_str_val = ['Video007','Video008','Video022']  # fold 5
# val_str_val = ['Video009','Video013','Video017']  # fold 6

#NEW THIS IS FOR VIDEO DIRECTORIES - NO FOLDS
video_dirs = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]

for video_dir in video_dirs:
    video_dir_path = os.path.join(TRAIN_DIR, video_dir)

    #collect all image + label files
    image_dir = os.path.join(video_dir_path, 'images')
    label_dir = os.path.join(video_dir_path, 'labels')

    image_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
    label_files = [os.path.join(label_dir, f) for f in sorted(os.listdir(label_dir))]

    image_paths.extend(image_files)
    label_paths.extend(label_files)

x_train_paths, x_valid_paths, y_train_paths, y_valid_paths = train_test_split(  
    image_paths, label_paths, test_size=0.2, random_state=42
)

for i in range(1):#range(len(train_str)):
       
    LR = 0.001
    print("Learning Rate -->", LR) 
    #val_str = 'Video009'#train_str[i]      
    #test_str = train_str[i]

    # print("Fold -->", val_str) 

    
    # x_train_dir = os.path.join(DATA_DIR+val_str+'_fold/train/images')
    # y_train_dir = os.path.join(DATA_DIR+val_str+'_fold/train/labels')
    
    # x_valid_dir = os.path.join(DATA_DIR+val_str+'_fold/val/images')
    # y_valid_dir = os.path.join(DATA_DIR+val_str+'_fold/val/labels')
    
    # x_test_dir = os.path.join(DATA_DIR+val_str+'_fold/val/images')
    # y_test_dir = os.path.join(DATA_DIR+val_str+'_fold/val/labels')

    #NEW list of paths for training and validating
    x_train_dir = x_train_paths
    y_train_dir = y_train_paths

    x_valid_dir = x_valid_paths
    y_valid_dir = y_valid_paths
    
    
    """
    #for getting fold data one time uncomment
    if MODE == 'train':  
        train_str1 = train_str.copy()
        for item in val_str_val:
            if item in train_str: 
                train_str1.remove(item)
        # Creating training data folder  
        if os.path.isdir(DATA_DIR+'/'+val_str+'_fold/train'):
            shutil.rmtree(DATA_DIR+'/'+val_str+'_fold/train')
        for i in range(len(train_str1)):
            copy_tree(DATA_DIR+train_str1[i]+'/images',x_train_dir)
        for i in range(len(train_str1)):
            copy_tree(DATA_DIR+train_str1[i]+'/labels',y_train_dir)    
            
        #validation
        for i in range(len(val_str_val)):
            copy_tree(DATA_DIR+val_str_val[i]+'/images',x_valid_dir)
        for i in range(len(val_str_val)):
            copy_tree(DATA_DIR+val_str_val[i]+'/labels',y_valid_dir)  
        #copy_tree(DATA_DIR+val_str+'/images',x_valid_dir)
        #copy_tree(DATA_DIR+val_str+'/labels',y_valid_dir)
        
        IMG_COUNT = len(os.listdir(x_train_dir))  
        IMG_COUNT_VAL = len(os.listdir(x_valid_dir))
    """
    
    

    """
    # For testing the data loading and augmenetation outputs
    dataset = DatasetCV(x_train_dir, y_train_dir, classes=CLASSES)
    #dataset = Dataset(x_train_dir, y_train_dir, classes=[8])
    
    image, mask = dataset[10] # get some sample
    #if len(CLASSES)==1:
    #    plot_image_n_label(image, mask.squeeze())
    #else:     
    plot_image_n_label(image,np.argmax(mask,axis=2))
    #visualize(
    #    image=image, 
    #    cars_mask=mask.squeeze(),
    #)
    
    #### Visualize resulted augmented images and masks
    augmented_dataset = DatasetCV(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(), 
        classes=CLASSES,
    )
    
    
    # same image with different random transforms
    for i in range(10):
        image, mask = augmented_dataset[i]
        #if len(CLASSES)==1:
        #    plot_image_n_label(image, mask.squeeze())
        #else:     
        plot_image_n_label(image,np.argmax(mask,axis=2))
        #visualize(image=image, mask=mask.squeeze())

    """
    # create segmentation model with pretrained encoder
    if ARCH == 'FPN':
        model = smp.FPN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    elif ARCH == 'DeepLabV3':
        model = smp.DeepLabV3(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    elif ARCH == 'Unet':
        model = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    elif ARCH == 'Unet++':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    elif ARCH == 'DeepLabV3+':
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )        
        
        
    #print(model)
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    #loss1 = smp.utils.losses.DiceLoss()
    #loss1 = smp.utils.losses.JaccardLoss()
    loss2 = smp.utils.losses.BCELoss()
    loss = loss2 #+ loss2
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5), #dont treshold for confidence maps
    ]

    if MODE == 'train':
    
        # For dataset handling
        train_dataset = DatasetCV(
            images_dir = x_train_paths, 
            masks_dir = y_train_paths, 
            augmentation=get_training_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )
        
        valid_dataset = DatasetCV(
            images_dir = x_valid_paths, 
            masks_dir = y_valid_paths, 
            #augmentation=get_validation_augmentation(), 
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )
        
        """        
    
        # check training and validation images and masks
        for i in range(3):
            image, mask = train_dataset[1]
            image = np.transpose(image,[1,2,0])
            visualize(image=image, mask=np.squeeze(mask))
        for i in range(3):
            image, mask = valid_dataset[1]
            image = np.transpose(image,[1,2,0])
            visualize(image=image, mask=np.squeeze(mask))
         """   

        assert len(x_train_paths) == len(y_train_paths), "Training images and labels length mismatch!"
        assert len(x_valid_paths) == len(y_valid_paths), "Validation images and labels length mismatch!"

        print(f"Number of training samples: {len(x_train_paths)}")
        print(f"Number of validation samples: {len(x_valid_paths)}") 
    
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=True)
            
        optimizer = torch.optim.Adam([ 
            dict(params=model.parameters(), lr=LR),
        ])
        
        # create epoch runners 
        # it is a simple loop of iterating over dataloader`s samples
        train_epoch = smp.utils.train.TrainEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )
        
        valid_epoch = smp.utils.train.ValidEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            device=DEVICE,
            verbose=True,
        )
        
        # train model for 'EPOCHS'
        max_score = 0
        
        for i in range(0, EPOCHS):
            
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            
            # do something (save model, change lr, etc.)
            if max_score < train_logs['iou_score']:
                max_score = train_logs['iou_score']
                CHECK_PATH = './checkpoints/best_'+ARCH+'_'+ENCODER + '_' + ACTIVATION + '_BCE_CLASSES'+str(len(class_values)) +'.pth'
                checkpoint_dir = os.path.dirname(CHECK_PATH)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                    print(f"Created directory: {checkpoint_dir}")
                torch.save(model, CHECK_PATH)
                print("Model saved to > ", CHECK_PATH)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
   
            if i == 100 or i==200:
                LR = LR/10
                optimizer.param_groups[0]['lr'] = LR
                print('Decrease decoder learning rate to LR/10!')
                
            # Check early stopping condition
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!' )
                early_stop = True
                break
            else:
                continue
            break
            if early_stop:
                print("Stopped")
                break
            
                
        del model

if MODE == 'predict':
    CHECK_PATH = './checkpoints/best_'+ARCH+'_'+ENCODER + '_' + ACTIVATION + '_BCE_CLASSES' + str(len(class_values)) +'.pth'
    output_path_vis = '../dataset/test/'+ARCH+'_'+ENCODER + '_' + ACTIVATION + '_BCE_CLASSES' + str(len(class_values)) + '/'

    vis_dir = os.path.join(output_path_vis, 'vis/')
    pred_mask_dir = os.path.join(output_path_vis, 'predicted_mask/')

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    if not os.path.exists(pred_mask_dir):
        os.makedirs(pred_mask_dir)

    print(f"Output path vis: {output_path_vis}")
    print(f"Vis directory: {vis_dir}")
    print(f"Predicted mask directory: {pred_mask_dir}")

    print(f"Directories created: {vis_dir}, {pred_mask_dir}")
        
    best_model = torch.load(CHECK_PATH)
    best_model.eval() 
    
    valid_dataset = DatasetCV(
        images_dir = x_valid_paths, 
        masks_dir = y_valid_paths, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    
    valid_dataset_vis = DatasetCV_test(
        images_dir = x_valid_paths, 
        classes=CLASSES,
    )
    
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)

    num_images = len(x_valid_paths) 
    
    with torch.no_grad():
        pa_list = []
        pac_list = []
        miou_2nd_list = []
        gt_mask_list = []
        pr_mask_list = []
        for i in range(num_images): 
            image_vis, temp, ids = valid_dataset_vis[i]
            image, gt_mask = valid_dataset[i]
            
            gt_mask = gt_mask.squeeze()
            
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            pr_mask = best_model(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
            pr_mask_c = np.argmax(pr_mask, axis=0)
            gt_mask_c = np.argmax(gt_mask, axis=0)
            
            image = (np.transpose(image, [1, 2, 0])).astype(float)
            
            plot_image_gt_pred_labels(image_vis, gt_mask_c, pr_mask_c)
            idspng = ids.replace('jpeg', 'png')
            plt.savefig(output_path_vis + 'vis/' + idspng)
            plt.close()
            
            result = cv2.imwrite(output_path_vis + '/predicted_mask/' + idspng, pr_mask_c)
            if result:
                print(output_path_vis + '/' + idspng + ' output mask saved')
            else:
                print('Error in saving file')
            
            gt_mask_list.append(gt_mask_c)
            pr_mask_list.append(pr_mask_c)
            
            accuracy, prec, rec, f1, miou = evaluate_segmentation(pred=pr_mask_c, label=gt_mask_c, num_classes=1)
            pa, pac, miou_2nd = segmentation_metrics(gt_mask_c, pr_mask_c, num_classes=len(class_values)+1)
            
            pa_list.append(pa)
            pac_list.append(pac)
            miou_2nd_list.append(miou_2nd)
                   
        avg_pa = np.mean(pa_list)
        std_pa = np.std(pa_list)
        
        avg_pac = np.mean(pac_list)
        std_pac = np.std(pac_list)
        
        avg_miou_2nd = np.mean(miou_2nd_list)
        std_miou_2nd = np.std(miou_2nd_list)              

        print("Average PA = ", avg_pa)
        print("Std PA = ", std_pa)

        print("Average PAC = ", avg_pac)
        print("Std PAC = ", std_pac)
        
        print("Average IoU 2nd = ", avg_miou_2nd)
        print("Std IoU 2nd = ", std_miou_2nd)

        # Convert lists to NumPy arrays before final evaluation
        gt_mask_array = np.array(gt_mask_list)
        pr_mask_array = np.array(pr_mask_list)

        accuracy, prec, rec, f1, miou = evaluate_segmentation(pred=pr_mask_array, label=gt_mask_array, num_classes=1)
        
        pa, pac, miou_2nd = segmentation_metrics(gt_mask_array, pr_mask_array, num_classes=len(class_values)+1)

        print("pa calculated overall = ", pa)
        print("pac calculated overall = ", pac)
        print("miou 2nd calculated overall = ", miou_2nd)
        print("accuracy = ", accuracy)
        print("precision = ", prec)
        print("recall = ", rec)
        print("F1 = ", f1)

          
    # if MODE == 'predict':
        
    #     CHECK_PATH = './checkpoints/best_'+ARCH+'_'+ENCODER + '_' + ACTIVATION + '_BCE_CLASSES' + str(len(class_values)) +'.pth'
    #     output_path_vis = '../dataset/test/'+ARCH+'_'+ENCODER + '_' + ACTIVATION + '_BCE_CLASSES' + str(len(class_values)) + '/'

    #     vis_dir = os.path.join(output_path_vis, 'vis/')
    #     pred_mask_dir = os.path.join(output_path_vis, 'predicted_mask/')

    #     if not os.path.exists(vis_dir):
    #         os.makedirs(vis_dir)
    #     if not os.path.exists(pred_mask_dir):
    #         os.makedirs(pred_mask_dir)

    #     print(f"Output path vis: {output_path_vis}")
    #     print(f"Vis directory: {vis_dir}")
    #     print(f"Predicted mask directory: {pred_mask_dir}")

    #     print(f"Directories created: {vis_dir}, {pred_mask_dir}")
            
    #     best_model = torch.load(CHECK_PATH)
    #     best_model.eval() 
        
    #     valid_dataset = DatasetCV(
    #         images_dir = x_valid_paths, 
    #         masks_dir = y_valid_paths, 
    #         #augmentation=get_validation_augmentation(), 
    #         preprocessing=get_preprocessing(preprocessing_fn),
    #         classes=CLASSES,
    #     )
        
    #     # validation dataset without transformations for image visualization
    #     valid_dataset_vis = DatasetCV_test(
    #         images_dir = x_valid_paths, 
    #         #y_valid_dir, 
    #         classes=CLASSES,
    #     )
        
    #     valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)

    #     # evaluate model on test set
    #     val_epoch = smp.utils.train.ValidEpoch(
    #         model=best_model,
    #         loss=loss,
    #         metrics=metrics,
    #         device=DEVICE,
    #     )
        
    #     #logs = val_epoch.run(valid_loader)        
        
        
    #     num_images = len(x_valid_paths) 
        
    #     with torch.no_grad():
    #         #prediction = torch.clamp(prediction, 0, 1)
    #         pa_list = []
    #         pac_list = []
    #         miou_2nd_list = []
    #         gt_mask_list = []
    #         pr_mask_list = []
    #         for i in range(num_images): 
    #             image_vis, temp, ids = valid_dataset_vis[i]
    #             image, gt_mask = valid_dataset[i]
                
    #             gt_mask = gt_mask.squeeze()
                
    #             x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    #             pr_mask = best_model(x_tensor)
    #             pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    #             #pr_mask = (pr_mask.squeeze().cpu().numpy())      
                
    #             pr_mask_c = np.argmax(pr_mask, axis = 0)
    #             gt_mask_c = np.argmax(gt_mask, axis = 0)
                
    #             pr_maskrgb = np.argmax(pr_mask, axis = 0)
    #             gt_maskrgb = np.argmax(gt_mask, axis = 0)
                
    #             #pr_mask_c = pr_mask.copy()
    #             #gt_mask_c = gt_mask.copy()
    #             #for j in range(len(class_values)):
    #             #    if len(class_values) == 1:
    #             #        pr_mask_c = pr_mask_c*class_values[j] 
    #             #        gt_mask_c = gt_mask_c*class_values[j] 
    #             #    else:
    #             #        pr_mask_c[j,:,:] = pr_mask_c[j,:,:]*class_values[j] 
    #             #        gt_mask_c[j,:,:] = gt_mask_c[j,:,:]*class_values[j] 
    #             #color2index, index2color = color_convCV()
    #             #gt_maskrgb = mask2rgb(gt_mask_c, index2color)
    #             #pr_maskrgb = mask2rgb(pr_mask_c, index2color)
                
    #             image = (np.transpose(image,[1,2,0])).astype(float)
                
    #             #gt_maskrgb= (np.transpose(gt_maskrgb,[1,2,0])).astype(np.uint8)
    #             #pr_maskrgb= (np.transpose(pr_maskrgb,[1,2,0])).astype(np.uint8) 
                
    #             plot_image_gt_pred_labels(image_vis, gt_mask_c, pr_mask_c)
    #             idspng = ids.replace('jpeg', 'png')
    #             plt.savefig(output_path_vis +'vis/'+idspng)
    #             plt.close()
                
    #             result=cv2.imwrite(output_path_vis +'/predicted_mask/'+idspng , pr_mask_c)
    #             if result:
    #                 print(output_path_vis + '/' + idspng + ' output mask saved')
    #             else:
    #                 print('Error in saving file')
                
    #             """
    #             visualize(
    #                 image=image, 
    #                 ground_truth_mask=gt_maskrgb, 
    #                 predicted_mask=pr_maskrgb
    #             )
    #             """
    #             gt_mask_list.append(gt_mask_c)
    #             pr_mask_list.append(pr_mask_c) 

    #             gt_mask_array = np.array(gt_mask_list)
    #             pr_mask_array = np.array(pr_mask_list)
    #             accuracy, prec, rec, f1, miou = evaluate_segmentation(pred=pr_mask, label=gt_mask, num_classes=1)
            
    #             pa, pac, miou_2nd = segmentation_metrics(gt_mask_c, pr_mask_c, num_classes = len(class_values)+1)
                                
    #             pa_list.append(pa)
    #             pac_list.append(pac)
    #             miou_2nd_list.append(miou_2nd)
                       
    #         avg_pa = np.mean(pa_list)
    #         std_pa = np.std(pa_list)
            
    #         avg_pac = np.mean(pac_list)
    #         std_pac = np.std(pac_list)
            
    #         avg_miou_2nd = np.mean(miou_2nd_list)
    #         std_miou_2nd = np.std(miou_2nd_list)                  
  
    #         print("Average PA = ", avg_pa)
    #         print("Std PA = ", std_pa)
    
    #         print("Average PAC = ", avg_pac)
    #         print("Std PAC = ", std_pac)
            
    #         print("Average IoU 2nd = ", avg_miou_2nd)
    #         print("Std IoU 2nd = ", std_miou_2nd)

    #         gt_mask_list.append(gt_mask_c)
    #         pr_mask_list.append(pr_mask_c) 


    #         gt_mask_array = np.array(gt_mask_list)
    #         pr_mask_array = np.array(pr_mask_list)

    #         accuracy, prec, rec, f1, miou = evaluate_segmentation(pred=pr_mask_list, label=gt_mask_list, num_classes=1)
            
    #         pa, pac, miou_2nd = segmentation_metrics(gt_mask_list, pr_mask_list, num_classes = len(class_values)+1)
        
        
    #         print("pa calculated overall = ", pa)
    #         print("pac calculated overall = ", pac)

    #         print("miou 2nd calculated overall = ", miou_2nd)


    '''        
    if MODE == 'test':

        
        CHECK_PATH = './checkpoints/best_'+ARCH+'_'+ENCODER + '_' + ACTIVATION + '_BCE_CLASSES' + str(len(class_values)) + '.pth'
        output_path_vis = '../dataset/test/'+ARCH+'_'+ENCODER + '_' + ACTIVATION + '_CLASSES' + str(len(class_values)) + '/vis/'

        #CHECK_PATH = './checkpoints/best_Unet++_resnet50_sigmoid_BCE_CLASSES4_All_fold.pth'

         # load best saved checkpoint
        #checkname = 'best_'+ARCH+'_'+ENCODER + '_' + ACTIVATION
        
        best_model = torch.load(CHECK_PATH)  
        best_model.eval()
        
        seqname = 'video1537'
        
        save_output_fig = True
    
        x_test_dir = '../dataset/'+seqname+'/images'
        outpath = '../dataset/'+seqname+'/predicted_mask'
        vispath = '../dataset/'+seqname+'/vis'
        
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        
        if save_output_fig and not os.path.isdir(vispath):
            os.makedirs(vispath)
        
        # Create test dataset
        test_dataset = DatasetCV_test(
            images_dir=x_test_dir, 
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )
    
        with torch.no_grad():
            num_images = len(test_dataset)
            for i in range(num_images):
                
                image, _, ids = test_dataset[i]  # Ground truth mask is ignored since it's not needed in test
                
                x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                pr_mask = best_model(x_tensor)
                pr_mask = pr_mask.squeeze().cpu().numpy() #confidence

                # Convert the prediction mask to a single-channel mask
                pr_mask_c = np.argmax(pr_mask, axis=0)  # Now pr_mask_c is (height, width)
                
                # Undo normalization for visualization
                image_vis = image.copy()
                image_vis = image_vis.transpose(1, 2, 0)  # Convert CHW to HWC for display
                image_vis = image_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Undo normalization
                image_vis = np.clip(image_vis, 0, 1)  # Clip to [0, 1] range

                # Optionally rescale for visualization
                pr_mask_viz = (pr_mask_c * 255 / pr_mask_c.max()).astype('uint8')

                # Visualize the results
                if save_output_fig:
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(image_vis)  # Image was originally in CHW format
                    plt.title('Input')

                    plt.subplot(1, 2, 2)
                    plt.imshow(pr_mask_viz, cmap='gray')  # pr_mask_viz is now 2D
                    plt.title('Predicted Mask')
                    
                    plt.savefig(os.path.join(vispath, ids))
                    plt.close()
                
                # Save the predicted mask
                pr_mask_c_resized = cv2.resize(pr_mask_c, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST) 
                result = cv2.imwrite(os.path.join(outpath, ids), pr_mask_c_resized)
                if result:
                    print(f'{outpath}/{ids} output mask saved')
                else:
                    print('Error in saving file')
    '''
    
if MODE == 'test':
    
    CHECK_PATH = './checkpoints/best_'+ARCH+'_'+ENCODER + '_' + ACTIVATION + '_BCE_CLASSES' + str(len(class_values)) + '.pth'
    output_path_vis = '../dataset/test/'+ARCH+'_'+ENCODER + '_' + ACTIVATION + '_CLASSES' + str(len(class_values)) + '/vis/'

    best_model = torch.load(CHECK_PATH)  
    best_model.eval()
    
    seqname = 'Video025'
    
    save_output_fig = True

    x_test_dir = '../dataset/'+seqname+'/images'
    outpath = '../dataset/'+seqname+'/predicted_mask'
    vispath = '../dataset/'+seqname+'/vis'
    
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    
    if save_output_fig and not os.path.isdir(vispath):
        os.makedirs(vispath)
    
    # Create test dataset
    test_dataset = DatasetCV_test(
        images_dir=x_test_dir, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    with torch.no_grad():
        num_images = len(test_dataset)
        for i in range(num_images):
            
            image, _, ids = test_dataset[i]
            
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            pr_mask = best_model(x_tensor)
            pr_mask = pr_mask.squeeze().cpu().numpy()  # Ensure this remains a float array, not rounded or binarized

            # Ensure no conversion to binary here
            vessel_confidence_map = pr_mask[1]  # Only select the vessel channel

            # Debugging: Print the min and max to check if it's binary or continuous
            #print(f"Min value: {vessel_confidence_map.min()}, Max value: {vessel_confidence_map.max()}")
            #print(f"Unique values: {np.unique(vessel_confidence_map)}")

            # Visualization
            plt.imshow(vessel_confidence_map, cmap='viridis')
            plt.title('Vessel Confidence Map')
            plt.colorbar()
            plt.show()

            # Save the vessel confidence map
            vessel_confidence_resized = cv2.resize(vessel_confidence_map, (image.shape[2], image.shape[1]), interpolation=cv2.INTER_LINEAR)
            result = cv2.imwrite(os.path.join(outpath, ids), (vessel_confidence_resized * 255).astype('uint8'))  # Save as 8-bit image
            if result:
                print(f'{outpath}/{ids} vessel confidence map saved')
            else:
                print('Error in saving file')
            plt.close()
