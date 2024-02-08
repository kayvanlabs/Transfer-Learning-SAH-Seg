import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
from sklearn.model_selection import KFold
import random

# tensorly decompositions
from tensorly.decomposition import parafac, tucker, tensor_train
import tensorly as tl
tl.set_backend('pytorch')

# Customized Files
from util.config import Config
from util.loss import SoftDiceLoss
from util.data.augment import DataAug
from util.data.loader import load_data
from util.models.network import get_model
from util.data.random_contrast import random_contrast
from util.evaluate import evaluate_old_data
from util.data.split import train_val_split

def get_args():
    parser = argparse.ArgumentParser(description='Train/tune a model')
    parser.add_argument('--tune', type=str, default='none', help='parts of model to tune')
    parser.add_argument('--name', type=str, default='', help='experiment name')
    parser.add_argument('--epoch_number', type=int, default=60, help='number of epochs to tune')
    parser.add_argument('--level', type=int, default=2, help='level of model')
    parser.add_argument('--decomposition', type=str, default='none', help='decomposition method')
    parser.add_argument('--trained_model_path', type=str, default='', help='path to pre-trained model')
    parser.add_argument('--train', action='store_true', help='train the model')
    args = parser.parse_args()
    return args

def compress_model(model, decomposition, modules):

    def run_cp(tensor, rank):
        cp_tensor = parafac(
            tensor=tensor,
            rank=rank,
            n_iter_max=100,
            tol=1e-6,
            linesearch=True
        )

        return tl.cp_to_tensor(cp_tensor)

    def run_tucker(tensor, rank):
        tucker_tensor = tucker(
        tensor,
        rank=rank,
        n_iter_max=100,
        tol=1e-6
        )

        return tl.tucker_to_tensor(tucker_tensor)

    def run_tt(tensor, rank):
        tt = tensor_train(
        tensor,
        rank=rank
        )

        return tl.tt_to_tensor(tt)

    if decomposition == 'cp':
        run_decomposition = run_cp
    elif decomposition == 'tucker':
        run_decomposition = run_tucker
    elif decomposition == 'tt':
        run_decomposition = run_tt       

    top = [
        'M1.DoubleConv.0.weight',
        'M1.DoubleConv.3.weight',
        'M3.TripleOut.0.weight',
        'M3.TripleOut.3.weight'
    ]

    left = [
        'M1.DoubleConv.0.weight',
        'M1.DoubleConv.3.weight',
        'down_list.0.M1.DoubleConv.0.weight',
        'down_list.0.M1.DoubleConv.3.weight',
        'down_list.1.M1.DoubleConv.0.weight',
        'down_list.1.M1.DoubleConv.3.weight'
    ]

    bottom = [
        'down_list.1.M1.DoubleConv.0.weight',
        'down_list.1.M1.DoubleConv.3.weight',
        'up_list.0.M1.DoubleConv.0.weight',
        'up_list.0.M1.DoubleConv.3.weight'
    ]

    right = [
        'up_list.0.M1.DoubleConv.0.weight',
        'up_list.0.M1.DoubleConv.3.weight',
        'up_list.1.M1.DoubleConv.0.weight',
        'up_list.1.M1.DoubleConv.3.weight',
        'M3.TripleOut.0.weight',
        'M3.TripleOut.3.weight'
    ]

    if modules == 'top':
        weight_names = top
    elif modules == 'left':
        weight_names = left
    elif modules == 'bottom':
        weight_names = bottom
    elif modules == 'right':
        weight_names = right

    max_rank = 100
    compressed_weights = {}

    # find the best rank for each set of weights
    for weight_name in weight_names:

        tensor = model.state_dict()[weight_name]
        error = 1
        rank = 0

        while error > 0.1 and rank < max_rank:
            rank += 1
            try:
                approx = run_decomposition(tensor, rank) # run decomposition
                error = (tl.norm(tensor - approx) / tl.norm(tensor)).item() # compute scaled reconstruction error
            except:
                pass

        compressed_weights[weight_name] = approx
        logging.info(f'{weight_name} rank: {rank} error: {round(error, 6)}')

    # replace weights in model with compressed weights
    if modules == 'top':
        with torch.no_grad():
            model.M1.DoubleConv[0].weight = torch.nn.parameter.Parameter(compressed_weights['M1.DoubleConv.0.weight'])
            model.M1.DoubleConv[3].weight = torch.nn.parameter.Parameter(compressed_weights['M1.DoubleConv.3.weight'])
            model.M3.TripleOut[0].weight = torch.nn.parameter.Parameter(compressed_weights['M3.TripleOut.0.weight'])
            model.M3.TripleOut[3].weight = torch.nn.parameter.Parameter(compressed_weights['M3.TripleOut.3.weight'])
    elif modules == 'left':
        with torch.no_grad():
            model.M1.DoubleConv[0].weight = torch.nn.parameter.Parameter(compressed_weights['M1.DoubleConv.0.weight'])
            model.M1.DoubleConv[3].weight = torch.nn.parameter.Parameter(compressed_weights['M1.DoubleConv.3.weight'])
            model.down_list[0].M1.DoubleConv[0].weight = torch.nn.parameter.Parameter(compressed_weights['down_list.0.M1.DoubleConv.0.weight'])
            model.down_list[0].M1.DoubleConv[3].weight = torch.nn.parameter.Parameter(compressed_weights['down_list.0.M1.DoubleConv.3.weight'])
            model.down_list[1].M1.DoubleConv[0].weight = torch.nn.parameter.Parameter(compressed_weights['down_list.1.M1.DoubleConv.0.weight'])
            model.down_list[1].M1.DoubleConv[3].weight = torch.nn.parameter.Parameter(compressed_weights['down_list.1.M1.DoubleConv.3.weight'])
    elif modules == 'bottom':
        with torch.no_grad():
            model.down_list[1].M1.DoubleConv[0].weight = torch.nn.parameter.Parameter(compressed_weights['down_list.1.M1.DoubleConv.0.weight'])
            model.down_list[1].M1.DoubleConv[3].weight = torch.nn.parameter.Parameter(compressed_weights['down_list.1.M1.DoubleConv.3.weight'])
            model.up_list[0].M1.DoubleConv[0].weight = torch.nn.parameter.Parameter(compressed_weights['up_list.0.M1.DoubleConv.0.weight'])
            model.up_list[0].M1.DoubleConv[3].weight = torch.nn.parameter.Parameter(compressed_weights['up_list.0.M1.DoubleConv.3.weight'])
    elif modules == 'right':
        with torch.no_grad():
            model.up_list[0].M1.DoubleConv[0].weight = torch.nn.parameter.Parameter(compressed_weights['up_list.0.M1.DoubleConv.0.weight'])
            model.up_list[0].M1.DoubleConv[3].weight = torch.nn.parameter.Parameter(compressed_weights['up_list.0.M1.DoubleConv.3.weight'])
            model.up_list[1].M1.DoubleConv[0].weight = torch.nn.parameter.Parameter(compressed_weights['up_list.1.M1.DoubleConv.0.weight'])
            model.up_list[1].M1.DoubleConv[3].weight = torch.nn.parameter.Parameter(compressed_weights['up_list.1.M1.DoubleConv.3.weight'])
            model.M3.TripleOut[0].weight = torch.nn.parameter.Parameter(compressed_weights['M3.TripleOut.0.weight'])
            model.M3.TripleOut[3].weight = torch.nn.parameter.Parameter(compressed_weights['M3.TripleOut.3.weight'])
    
    return model

def freeze_layers(model, modules):
    '''
    Freeze all children besides those whose index is in the tune_children list.
    '''

    if modules == 'all':
        logging.info(f"Tuning all layers")
        return model

    # freeze all weights
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze the ones we want to tune
    if modules == 'left':
        tune = [model.M1, model.down_list[0].M1, model.down_list[1].M1] # left, down
        logging.info(f"Tuning left-side M1 modules")
    elif modules == 'right':
        tune = [model.up_list[0].M1, model.up_list[1].M1, model.M3] # right, up
        logging.info(f"Tuning right-side M1 modules and M3")
    elif modules == 'top':
        tune = [model.M1, model.M3]
        logging.info(f"Tuning top M1 and M3 modules")
    elif modules == 'bottom':
        tune = [model.down_list[1].M1, model.up_list[0].M1]
        logging.info(f"Tuning bottom M1 modules")
    elif modules == 'none':
        logging.info('Freezing all layers')
        return model

    for layer in tune:
        for param in layer.parameters():
            param.requires_grad = True

    return model

def make_dirs(exp_name):

    # create output directory
    dir_output = './outputs/'
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)
    
    # create experiment output directory
    dir_exp = './outputs/{}'.format(exp_name)
    if not os.path.isdir(dir_exp):
        os.mkdir(dir_exp)

    # create check point directory
    dir_checkpoint = '{}/checkpoints/'.format(dir_exp)
    if not os.path.isdir(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    return dir_output, dir_exp, dir_checkpoint

def mix(tune_set):
    '''
    Shuffle and order the tuning set so each batch will have one positive and one negative sample.
    '''

    random.seed(0)
    
    positive_data = []
    negative_data = []

    for patient in tune_set:
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
    for i in range(len(positive_data)):
        mix_data.append(positive_data[i])
        if i < len(negative_data):
            mix_data.append(negative_data[i])

    return mix_data

def tune(model, device, dir_checkpoint, dir_exp, epoch_number, tune, tune_set, fold):

    number_tune = len(tune_set)
    
    # perform data augmentation (elastic transform, horizontal flip)
    tune_set = DataAug(tune_set)

    # create data loader
    tune_loader_args = dict(drop_last = True, shuffle = False, batch_size = config.batch_size, 
                       num_workers = 0, pin_memory = True)
    tune_loader = DataLoader(tune_set, **tune_loader_args)

    # The original paper uses adam with learning rate of 0.001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = config.learning_rate)
    
    # proposed mixed loss
    criterion = SoftDiceLoss()

    # logging the tuning configuration
    logging.info(f'''Starting tuning:
        Model:           {model.__class__.__name__}
        Epochs:          {epoch_number}
        Batch size:      {config.batch_size}
        Learning rate:   {config.learning_rate}
        Tuning:          {tune}
        Fold:            {fold}
    ''')

    # remember how many step total has been taken
    global_step = 0

    max_tuning_dice = 0
    max_tuning_dice_checkpoint = None

    loss_list = []
    tuning_dice_list = []
    tune_pids = set()

    # tune
    for epoch in range(1, epoch_number+1):
        
        running_loss = 0
        running_dice = 0
        
        # start the tuning
        model.train()

        with tqdm(total=number_tune, desc=f'Epoch {epoch}/{epoch_number}', unit=' img') as pbar:

            # for every batch of images
            for i, (images, masks, pid) in enumerate(tune_loader):

                # random contrast
                images, weight = random_contrast(images, False, 30)
                
                # move the images to gpu or cpu
                images = images.to(device)
                masks = masks.to(device)
                weight = weight.to(device)
                
                # add zero grad
                optimizer.zero_grad()

                # get the prediction
                prediction = model(images.float())

                # calculate loss
                mix_loss = criterion(prediction, masks, weight)
                dice_loss = criterion(prediction, masks, torch.tensor([1.0]).to(device)) # this computes dice as if the entire batch is 1 image
                
                # gradient descent
                mix_loss.backward()        
                optimizer.step()
                global_step += 1
                
                # update the tqdm pbar
                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': mix_loss.item()})
                
                # calculating average loss dice
                running_loss += mix_loss.item()
                running_dice += 1 - dice_loss.item()

                tune_pids.add(pid[0].item())
                tune_pids.add(pid[1].item())

        # print loss
        logging.info('Average tuning loss is {} ; dice is {} ; step {} ; epoch {}.'.format(running_loss / len(tune_loader), running_dice / len(tune_loader), global_step, epoch))
        
        # for plotting dice and loss
        loss_list.append(running_loss / len(tune_loader))
        tuning_dice_list.append(running_dice / len(tune_loader))

        # save the best performing model
        if (running_dice / len(tune_loader)) >= max_tuning_dice:
            max_tuning_dice_checkpoint = model.state_dict().copy()
            max_tuning_dice = running_dice / len(tune_loader)

    # save checkpoints
    torch.save(max_tuning_dice_checkpoint, os.path.join(dir_checkpoint,'Max_Tuning_Dice_{}_fold-{}.pt'.format(max_tuning_dice, fold)))
    torch.save(model.state_dict(), os.path.join(dir_checkpoint,f'COMPLETED_TUNED_FOLD-{fold}.pt'))

    logging.info('Tuning Completed Model Saved')
    logging.info('Max Tuning Dice is {}'.format(max_tuning_dice))

    x = np.linspace(1, len(tuning_dice_list), len(tuning_dice_list))
    plt.figure()
    plt.plot(x, tuning_dice_list)
    plt.xlabel('Steps')
    plt.ylabel('Dice Score')
    plt.title('Tuning Dice')
    plt.savefig('{}/TuningDice_fold-{}.png'.format(dir_exp, fold))
    plt.close()

    plt.figure()
    plt.plot(x, loss_list)
    plt.xlabel('Steps')
    plt.ylabel('Mixed Loss')
    plt.title('Tuning Loss')
    plt.savefig('{}/TuningLoss_fold-{}.png'.format(dir_exp, fold))
    plt.close()

def dices_to_csv(dices, fold, path):
    '''
    Save the dices to a csv file.
    '''
    import csv
    with open(f'{path}/dices_fold-{fold}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Dice', 'Mask_Pixel_Count', 'Pred_Pixel_Count', 'Patient'])
        for dice, mask_pixels, pred_pixels, pid in dices:
            writer.writerow([dice, mask_pixels, pred_pixels, pid])

def test(model, device, dir_exp, test_set, fold):

    logging.info(f'Testing on fold {fold}...')

    # Set model to evaluation mode
    model.eval()

    # Initialize the Dice Loss
    scorer = SoftDiceLoss()
    dices = []
    slice_dices = []

    # create image directory
    dir_img = '{}/result_img_newdata_{}/'.format(dir_exp, fold)
    if not os.path.isdir(dir_img):
        os.mkdir(dir_img)

    # For every patient in the dataset
    for patient in test_set:

        # create test loader for each patients
        test_loader_args = dict(drop_last = False, shuffle = False, batch_size = 1, 
                num_workers = 0, pin_memory = True)
        test_loader = DataLoader(patient, **test_loader_args)

        # store mask and prediction per patients
        predictions = []
        masks = []

        # for every slices in the test loader
        for i, (images, mask, pid, condition) in enumerate(test_loader):

            # make patient dir
            if i == 0:
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
                per_image_dice = round(1 - scorer(prediction.cpu(), mask.cpu(), 1).item(), 3)
            
                # before plot the image we need go through sigmoid
                prediction = torch.sigmoid(prediction)
            
            pixel = mask[0].flatten().sum()
            pred_pixel = prediction[0][0].flatten().sum()
            slice_dices.append([per_image_dice, pixel.item(), pred_pixel.item(), pid.item()])

            # I only plotted the positive masks
            if pixel > 0:

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

        patient_dice = 1 - scorer(torch.stack(predictions), torch.stack(masks), 1).item()

        logging.info("Dice Score for Patient {} is {}".format(pid.item(), patient_dice))
        dices.append(patient_dice)
    
    logging.info('Average Dice Score for New Data is {}, standard deviation: {}'.format(sum(dices)/len(dices), torch.std(torch.tensor(dices))))
    dices_to_csv(slice_dices, fold, dir_img)

def train(model, device, dir_checkpoint, dir_exp, fix_split, epoch_number):

    # 1. Read data from matlab file
    data, patient_condition = load_data(config.load_num, config.training_directory)
    logging.info(f"Loaded {len(patient_condition)} Patients")

    # 2. Train eval Split
    train_set, val_set, test_set = train_val_split(data, patient_condition, 0.1, 0.0, fix_split)
    number_train = len(train_set)
    
    # perform data augmentation (elastic transform, horizontal flip)
    train_set = DataAug(train_set)

    # create data loader
    train_loader_args = dict(drop_last = True, shuffle = False, batch_size = config.batch_size, 
                       num_workers = 0, pin_memory = True)
    train_loader = DataLoader(train_set, **train_loader_args)

    # The original paper uses adam with learning rate of 0.001
    optimizer = optim.Adam(model.parameters(), lr = config.learning_rate)
    
    # proposed mixed loss
    criterion = SoftDiceLoss()

    # logging the training configuration
    logging.info(f'''Starting training:
        Model:           {model.__class__.__name__}
        Epochs:          {epoch_number}
        Batch size:      {config.batch_size}
        Learning rate:   {config.learning_rate}
        Training size:   {number_train}
    ''')

    # remember how many step total has been taken
    global_step = 0

    max_training_dice = 0
    max_testing_dice = 0
    max_training_dice_checkpoint = None
    max_testing_dice_checkpooint = None

    loss_list = []
    validation_dice_list = []
    training_dice_list = []

    # train for 200 epochs
    for epoch in range(1, epoch_number+1):
        
        running_loss = 0
        running_dice = 0
        
        # start the training
        model.train()

        with tqdm(total = number_train, desc = f'Epoch {epoch}/{epoch_number}', unit = ' img') as pbar:

            # for every batch of images
            for i, (images, masks, pid) in enumerate(train_loader):

                # random contrast
                images, weight = random_contrast(images, False, 30)
                
                # move the images to gpu or cpu
                images = images.to(device)
                masks = masks.to(device)
                weight = weight.to(device)
                
                # add zero grad
                optimizer.zero_grad()

                # get the prediction
                prediction = model(images.float())

                # calculate loss
                mix_loss = criterion(prediction, masks, weight)
                dice_loss = criterion(prediction, masks, torch.tensor([1.0]).to(device))
                
                # gradient descent
                mix_loss.backward()                
                optimizer.step()
                global_step += 1
                
                # update the tqdm pbar
                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': mix_loss.item()})
                
                # calculating average loss dice
                running_loss += mix_loss.item()
                running_dice += 1 - dice_loss.item()


        # print loss
        logging.info('Average train loss is {} ; dice is {} ; step {} ; epoch {}.'.format(running_loss / len(train_loader), running_dice / len(train_loader), global_step, epoch))

        # get evaluation score
        val_score = evaluate_old_data(model, val_set, device)

        # for plotting validation dice
        validation_dice_list.append(val_score)
        # for plotting dice and loss
        loss_list.append(running_loss / len(train_loader))
        training_dice_list.append(running_dice / len(train_loader))

        # log the evaluation score
        logging.info(f'Validation Average Subject Dice Score is {val_score}')


        # saving the best performed ones
        if (running_dice / len(train_loader)) >= max_training_dice:
            max_training_dice_checkpoint = model.state_dict().copy()
            max_training_dice = running_dice / len(train_loader)
        if val_score >= max_testing_dice:
            max_testing_dice = val_score
            max_testing_dice_checkpooint = model.state_dict().copy()


    # save the max checkpoints
    torch.save(max_training_dice_checkpoint, os.path.join(dir_checkpoint,'Max_Training_Dice_{}.pt'.format(max_training_dice)))
    torch.save(max_testing_dice_checkpooint, os.path.join(dir_checkpoint,'Max_Testing_Dice_{}.pt'.format(max_testing_dice)))
    # save the model after job is done
    torch.save(model.state_dict(), os.path.join(dir_checkpoint,'COMPLETED.pt'))


    logging.info('Training Completed Model Saved')
    logging.info('Max Training Dice is {}'.format(max_training_dice))
    logging.info('Max Testing Dice is {}'.format(max_testing_dice))

    x = np.linspace(1, len(training_dice_list), len(training_dice_list))
    plt.figure()
    plt.plot(x, training_dice_list)
    plt.xlabel('Steps')
    plt.ylabel('Dice Score')
    plt.title('Training Dice')
    plt.savefig('{}/TrainingDice.png'.format(dir_exp))
    plt.close()

    plt.figure()
    plt.plot(x, loss_list)
    plt.xlabel('Steps')
    plt.ylabel('Mixed Loss')
    plt.title('Training Loss')
    plt.savefig('{}/TrainingLoss.png'.format(dir_exp))
    plt.close()

    y = np.linspace(1, len(validation_dice_list), len(validation_dice_list))
    plt.figure()
    plt.plot(y, validation_dice_list)
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.title('Testing Dice')
    plt.savefig('{}/TestingDice.png'.format(dir_exp))
    plt.close()

    return model

if __name__ == '__main__':

    # get configurations and arguments
    config = Config()
    args = get_args()

    # create experiment name and directories
    exp_name = args.tune + '_' + args.decomposition + '_' + args.name + '_' + config.exp_name
    dir_output, dir_exp, dir_checkpoint = make_dirs(exp_name)

    # initialize the logging
    logging.basicConfig(filename='{}/Running.log'.format(dir_exp), level=logging.INFO, format='%(asctime)s: %(message)s')
    
    # get GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log which device we are using
    logging.info(f"Model is Running on {device}.")

    # load pre-trained model and freeze layers
    model = get_model(config.model_type, device, args.level)

    if args.train:
        model = train(model, device, dir_checkpoint, dir_exp, config.fix_split, args.epoch_number)
    else:
        model.load_state_dict(torch.load(args.trained_model_path))

    if args.decomposition != 'none':
        model = compress_model(model, args.decomposition, args.tune)

    if args.tune != 'none':
        model = freeze_layers(model, args.tune)

    # load data
    data, patient_condition = load_data(3, config.all_new_data_directory)
    logging.info(f"Loaded {len(patient_condition)} Patients")

    # split data by patient
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    for i, (tune_index, test_index) in enumerate(kf.split(data)):
        logging.info(f'Fold {i}:')
        tune_set = [data[i] for i in tune_index]
        test_set = [data[i] for i in test_index]

        # order tuning data into batches with positive and negative samples
        tune_set = mix(tune_set)

        # load pre-trained model without previous fold's tuning
        if not args.train:
            model.load_state_dict(torch.load(args.trained_model_path))

        if args.tune != 'none':
            # tune the model
            tune(model, device, dir_checkpoint, dir_exp, args.epoch_number, args.tune, tune_set, i)

        # evaluate the model
        test(model, device, dir_exp, test_set, i)