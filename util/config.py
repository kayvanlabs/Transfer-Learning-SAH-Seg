# Author: Kyle Ma @ BCIL 
# Created: 05/12/2023
# Implementation of Automated Hematoma Segmentation


class Config():

    # Experiment Congigurations
    epoch_number = 20
    epoch_tune_number = 20
    batch_size = 2
    learning_rate = 0.001
    eval_ratio = 0.1
    test_ratio = 0.3
    
    model_level = 2
    fix_split = True
    model_type = "MV" # MV
    # Unet, RRUnet, ResUnet, DRUnet, AttUnet, AttRRUnet, SEUnet
    # MV, MV_SE, MV_SE_RES, MV_F, MV_eval, Burger, TensorlyUnet, TensorlySEUnet, Burger_TCL, Burger_Plus, TLUnet

    # Test the new data or not
    test_new = True
    img_ct = True

    # Data Loading Configurations
    load_num = 5
    training_directory = "/path/to/training/data/"
    new_data_directory = "/path/to/new/data/"
    all_new_data_directory = "/path/to/all/new/data/"
    new_data_testing_directory = "/path/to/new/test/data/"

    # configurations for tuning a pre-trained model
    tune = True

    # Experiment Output Folder Name
    exp_series = 'fixed-cv'

    if model_type[0] == "M":
        if fix_split:
            exp_name = exp_series +'_locksplit_' + model_type + '_level' + str(model_level) + "_epoch" + str(epoch_number) + "_lr" + str(learning_rate)
        else:
            exp_name = exp_series + '_' + model_type + '_level' + str(model_level) + "_epoch" + str(epoch_number) + "_lr" + str(learning_rate)
    else:
        if fix_split:
            exp_name = exp_series +'_locksplit_' + model_type  + "_epoch" + str(epoch_number) + "_lr" + str(learning_rate)
        else:
            exp_name = exp_series + '_' + model_type + "_epoch" + str(epoch_number) + "_lr" + str(learning_rate)
            
trained_model_path = '/path/to/trained_model_state_dict.pt'
