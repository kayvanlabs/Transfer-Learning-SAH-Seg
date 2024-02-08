# Author: Kyle Ma @ BCIL 
# Created: 04/28/2023
# Implementation of Automated Hematoma Segmentation

import torch
import numpy as np
from util.loss import SoftDiceLoss
from torch.utils.data import DataLoader
from util.data.random_contrast import random_contrast
import logging
import os
import matplotlib.pyplot as plt


def evaluate(model, val_set, device):
    # start the evaluation mode
    model.eval()

    scorer = SoftDiceLoss()

    # contain total scores
    dice_score = 0
    pids = []

    for patient in val_set:

        # create test loader for each patients
        test_loader_args = dict(drop_last = False, shuffle = False, batch_size = 1, 
                   num_workers = 0, pin_memory = True)
        test_loader = DataLoader(patient, **test_loader_args)

        # store mask and prediction per patients
        predictions = []
        masks = []
    
        # for every slices in the test loader
        for i, (images, mask, pid, condition) in enumerate(test_loader):

            #make it 4D with batch size = 1
            images = images.unsqueeze(1)

            images, weight = random_contrast(images, False, 30)
            
            # move the images to gpu or cpu
            images = images.to(device)
            weight = weight.to(device)

            # get our prediction
            with torch.no_grad():
                prediction = model(images.float())

                predictions.append(prediction.cpu())
                masks.append(mask.cpu())

        pids.append(pid.item())
        dice_score += (1 - scorer(torch.stack(predictions), torch.stack(masks), 1).item())


    model.train()
    
    return dice_score / len(val_set), pids


def eval_and_plot(model, test_set, device, dir_exp, config):
    
    logging.info("Evaluating on Testing Data")

    # Set model to evaluation mode
    model.eval()

    # Initialize the Dice Loss
    scorer = SoftDiceLoss()
    dices = []

    # Counts the patients in the dataset
    j = 0

    # create image directory
    dir_img = '{}/result_img_newdata/'.format(dir_exp)
    if not os.path.isdir(dir_img):
        os.mkdir(dir_img)

    # For every patient in the dataset
    for patient in test_set:
        
        # Count number of patients
        j += 1
        
        # create test loader for each patients
        test_loader_args = dict(drop_last = False, shuffle = False, batch_size = 1, 
                num_workers = 0, pin_memory = True)
        test_loader = DataLoader(patient, **test_loader_args)

        # store mask and prediction per patients
        predictions = []
        masks = []

        # for every slices in the test loader
        for i, (images, mask, pid, condition) in enumerate(test_loader):

            if i == 0:
                # Each patient should have their own folder
                dir_img_per_patient = os.path.join(dir_img,'Patient{}'.format(pid.item()))
                if not os.path.isdir(dir_img_per_patient):
                    os.makedirs(dir_img_per_patient)

            # make it 4D with batch size = 1
            images = images.unsqueeze(1)

            # add normal contrasts
            images, weight = random_contrast(images, False, 30)

            # move the images to gpu or cpu
            images = images.to(device)
            weight = weight.to(device)

            # get our prediction
            with torch.no_grad():
                prediction = model(images.float())

                # store prediction and masks per patients
                predictions.append(prediction.cpu())
                masks.append(mask.cpu())
                
                # calculate per image dice
                per_image_dice = round(1 - scorer(prediction.cpu(), mask.cpu(), 1).item(), 2)
            
                # before plot the image we need go through sigmoid
                prediction = torch.sigmoid(prediction)
            

            pixel = mask[0].flatten().sum()

            # I only plotted the positive masks
            if pixel > 0:

                # do you want to plot the ct scan as well?
                if config.img_ct:
                    fig, (ax0, ax1, ax2) = plt.subplots(1,3)
                    ax0.imshow(images.cpu()[0][0], cmap='Greys_r')
                    ax0.set_title('CT Scan')
                    ax0.set_xticks([])
                    ax0.set_yticks([])
                    ax1.imshow(mask[0], cmap='Greys')
                    ax1.set_title('GroundTruth')
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    ax1.set_xlabel("Pixel {}".format(pixel))
                    ax2.imshow(prediction.cpu()[0][0], cmap='Greys')
                    ax2.set_title('Prediction')
                    ax2.set_xlabel("Dice {}".format(per_image_dice))
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    fig.savefig(os.path.join(dir_img_per_patient,'slice{}_{}_{}.png'.format(i, pixel, per_image_dice)), dpi=600)
                    plt.close()
                else:
                    fig, (ax1, ax2) = plt.subplots(1,2)
                    ax1.imshow(mask[0], cmap='Greys')
                    ax1.set_title('GroundTruth')
                    ax1.set_xticks([])
                    ax1.set_yticks([])
                    ax1.set_xlabel("Pixel {}".format(pixel))
                    ax2.imshow(prediction.cpu()[0][0], cmap='Greys')
                    ax2.set_title('Prediction')
                    ax2.set_xlabel("Dice {}".format(per_image_dice))
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    fig.savefig(os.path.join(dir_img_per_patient,'slice{}_{}_{}.png'.format(i, pixel, per_image_dice)), dpi=600)
                    plt.close()


        patient_dice = 1 - scorer(torch.stack(predictions), torch.stack(masks), 1).item()

        logging.info("Dice Score for Patient {}(#{}) is {}".format(pid.item(), j, patient_dice))
        dices.append(patient_dice)
    
    logging.info('Average Dice Score for New Data is {}, standard deviation: {}'.format(sum(dices)/len(dices), torch.std(torch.tensor(dices))))


def evaluate_old_data(model, val_set, device):
    # start the evaluation mode
    model.eval()

    scorer = SoftDiceLoss()

    # contain total scores
    dice_score = 0

    for patient in val_set:

        # create test loader for each patients
        test_loader_args = dict(drop_last = False, shuffle = False, batch_size = 1, 
                   num_workers = 0, pin_memory = True)
        test_loader = DataLoader(patient, **test_loader_args)

        # store mask and prediction per patients
        predictions = []
        masks = []
    
        # for every slices in the test loader
        for i, (images, mask, condition) in enumerate(test_loader):

            #make it 4D with batch size = 1
            images = images.unsqueeze(1)

            images, weight = random_contrast(images, False, 30)
            
            # move the images to gpu or cpu
            images = images.to(device)
            weight = weight.to(device)

            # get our prediction
            with torch.no_grad():
                prediction = model(images.float())

                predictions.append(prediction.cpu())
                masks.append(mask.cpu())

        dice_score += (1 - scorer(torch.stack(predictions), torch.stack(masks), 1).item())


    model.train()
    
    return dice_score / len(val_set)