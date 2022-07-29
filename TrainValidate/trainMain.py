# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:15:28 2022

@author: Laurens Kreilinger
"""
from kale.loaddata.videos import VideoFrameDataset
from kale.prepdata.video_transform import ImglistToTensor
import os
from torchvision import transforms
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
# local Packages
from TrainValidate import PhysNet
from TrainValidate import lossFunction
from TrainValidate import trainOneEpoch
from TrainValidate import get_pulse



def train_model(outputDataUBFCPath: str, Plot_results: bool,training_loader, validation_loader, test_loader):
    """

    :type outputDataUBFCPath: str
    :type Plot_results: bool
    """
    model_root = os.path.dirname(outputDataUBFCPath)
    model_root = os.path.join(model_root + '\\Model')
    loader_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #%% Define parameter for training
    EPOCHS = 5
    best_vloss = 1_000_000.
    learningRate = 0.001
    Plot_results = True

    model = PhysNet.PhysNet_padding_Encoder_Decoder_MAX(frames=128)
    model.to(loader_device)
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    loss_Inst = lossFunction.Neg_Pearson()
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    #%% train and validate modell
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        print('EPOCH {}:'.format(epoch_number + 1))

        # gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = trainOneEpoch.train_one_epoch(epoch_number, writer, training_loader, optimizer, model, loss_Inst)

        # gradient tracking of
        model.train(False)

        running_vloss = 0.0

        # Check model with validation data
        for i, vdata in enumerate(validation_loader):
            vinputs, BVP_vlabel = vdata
            # prepare data
            vinputs = vinputs.permute(0, 2, 1, 3, 4)  # [batch,channel,length,width,height] = x.shape
            # print(inputs.shape)
            BVP_label = torch.stack(BVP_vlabel)
            if torch.cuda.is_available() == True:
                vinputs = vinputs.cuda()
            rPPG, x_visual, x_visual3232, x_visual1616 = model(vinputs)
            if torch.cuda.is_available() == True:
                rPPG = rPPG.cpu()
            # Calculate the loss
            rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
            BVP_label = (BVP_label - torch.mean(BVP_label.float())) / torch.std(BVP_label.float())  # normalize
            loss_ecg = loss_Inst(rPPG, BVP_label)
            loss_ecg.backward()
            running_vloss += loss_ecg

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()
        # Plot
        if Plot_results:
            fps = 30
            rPPGNP = rPPG.detach().numpy()
            rPPGNP = np.transpose(rPPGNP)
            BVP_labelNP = BVP_label.detach().numpy()
            pulse_BVP_labelNP = get_pulse.get_rfft_pulse(BVP_labelNP, fps)  # get pulse from signal
            pulse_PPGNP = get_pulse.get_rfft_pulse(rPPGNP, fps)  # get pulse from signal
            max_time = rPPGNP.size / fps
            time_steps = np.linspace(0, max_time, rPPGNP.size)
            plt.figure(figsize=(15, 15))
            plt.title('EPOCH {}:'.format(epoch_number + 1))
            plt.plot(time_steps, rPPGNP, label='rPPG')
            plt.plot(time_steps, BVP_labelNP, label='BVP_label')
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.show()
            print('Label Puls {} Valid result {}'.format(pulse_BVP_labelNP, pulse_PPGNP))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = '\\model_{}_{}'.format(timestamp, epoch_number)
            model_path = os.path.join(model_root + model_path)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    return test_loader
    print('Finished Training')
