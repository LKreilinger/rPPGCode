# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:15:28 2022

@author: Laurens Kreilinger
"""
import os
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import wandb
# local Packages
from cnn_process.TrainValidate import trainOneEpoch, get_pulse
from cnn_process.load import lossFunction
from cnn_process.load.load_main import PhysNet


def train_and_validate_model(model, train_loader, validation_loader, loss_Inst, optimizer, config):
    """

    """



    Plot_results = True
    wandb.watch(model, loss_Inst, log="all", log_freq=10)

    #%% train and validate model
    best_vloss = 1_000_000.
    epoch_number = 0
    for epoch in range(config.epochs):  # loop over the dataset multiple times

        print('EPOCH {}:'.format(epoch_number + 1))

        # gradient tracking is on, and do a pass over the data
        model.train(True)
        ########################
        example_ct = 0  # number of examples seen
        for batch_ct, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            inputs, BVP_label = data

            loss_ecg = trainOneEpoch.train_batch(inputs, BVP_label, optimizer, model, loss_Inst)

            # Gather data and report
            example_ct += len(inputs)
            if batch_ct % 10 == 9:
                wandb.log({"epoch": epoch, "loss": loss_ecg}, step=example_ct)
                print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss_ecg:.3f}")

        ##################################
        # gradient tracking of
        model.train(False)

        # Check model with validation data
        example_validation_ct = 0  # number of examples seen
        running_vloss = 0.0
        for batch_validation_ct, vdata in enumerate(validation_loader):
            vinputs, BVP_vlabel = vdata
            # prepare data
            vinputs = vinputs.permute(0, 2, 1, 3, 4)  # [batch,channel,length,width,height] = x.shape
            # print(inputs.shape)
            BVP_label = torch.stack(BVP_vlabel)
            if torch.cuda.is_available():
                vinputs = vinputs.cuda()
            rPPG, x_visual, x_visual3232, x_visual1616 = model(vinputs)
            if torch.cuda.is_available():
                rPPG = rPPG.cpu()
            # Calculate the loss
            rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
            BVP_label = (BVP_label - torch.mean(BVP_label.float())) / torch.std(BVP_label.float())  # normalize
            loss_ecg = loss_Inst(rPPG, BVP_label)
            loss_ecg.backward()
            example_ct += len(vinputs)
            if batch_validation_ct % 10 == 9:
                wandb.log({"epoch": epoch, "loss": loss_ecg}, step=example_ct)
                print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss_ecg:.3f}")



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


        epoch_number += 1

    # save last model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = "model_{}_{}".format(timestamp, epoch_number)
    model_path = os.path.join(config.path_model, model_name)
    torch.save(model.state_dict(), model_path)
    print('Finished Training')
