# Author: Kyle Ma @ BCIL 
# Created: 05/02/2023
# Implementation of Automated Hematoma Segmentation

import random
import logging
import numpy as np

def train_val_split(data, patient_condition, val_ratio, test_ratio, fix_split):

    if fix_split:
        random.seed(727)

    data = np.array(data, dtype=object)

    # get the number of evaluation and training
    number_eval = int(len(data) * (val_ratio + test_ratio))
    number_val = int(len(data) * val_ratio)
    number_train = len(data) - number_eval

    # evaluation set only keeps unhealthy patients (meaning slices with hematoma annotation)
    val_test_temp = data[np.logical_not(patient_condition)]
    # training set can have healthy patients (meaning slices without hematoma annotation)
    train_temp = data[patient_condition]

    # get validation set random patient
    random.shuffle(val_test_temp)
    eval_set = val_test_temp[:number_eval]
    val_set = eval_set[:number_val]
    test_set = eval_set[number_val:]

    # get back our train set with healthy patients
    train_set = np.concatenate((val_test_temp[number_eval:],train_temp))

    # separate positive and negative data for training
    positive_data = []
    negative_data = []
    train_pids = []

    for patient in train_set:
        for slices in patient:
            if slices[-1] == 1:
                positive_data.append(slices[:3])
            else:
                negative_data.append(slices[:3])

    # shuffle to random order
    random.shuffle(positive_data)
    random.shuffle(negative_data)
    
    # data balancing method
    mix_data = []
    # min_len = min(len(positive_data), len(negative_data)) # get the minimum length
    for i in range(len(positive_data)):
        mix_data.append(positive_data[i])
        if i < len(negative_data):
            mix_data.append(negative_data[i])
        
    logging.info(f"There are {number_train} Patients for Training ({len(mix_data)} images)")
    logging.info(f"There are {len(val_set)} Patients for Validation")
    logging.info(f"There are {len(test_set)} Patients for Testing")

        
    return mix_data, val_set, test_set