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
from cnn_process.TrainValidate import train_batch, get_pulse, validate_batch


def train_and_validate_model(model, train_loader, validation_loader, loss_Inst, optimizer, config):
    """

    """

    Plot_results = False
    wandb.watch(model, loss_Inst, log="all", log_freq=10)
    print("cuda memory befor training")
    torch.cuda.memory_summary(device=None, abbreviated=False)
    # %% train and validate model
    epoch_number = 0.
    example_ct = 0.  # number of examples seen
    example_ct_validation = 0
    torch.cuda.empty_cache()
    for epoch in range(config.epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        running_loss = 0.
        last_loss = 0.
        model.train(True)
        for batch_ct, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            inputs, BVP_label = data
            loss = train_batch.train_batch(inputs, BVP_label, optimizer, model, loss_Inst)
            running_loss += loss.item()

            print("cuda memory after batch {}".format(batch_ct))
            torch.cuda.memory_summary(device=None, abbreviated=False)
            # Gather data and report
            example_ct += len(inputs)
            if batch_ct % 10 == 9:
                last_loss = running_loss / 10
                wandb.log({"epoch": epoch, "loss": last_loss})
                print('  batch {} loss: {}'.format(batch_ct + 1, last_loss))
                running_loss = 0.

        ##################################
        # Validate model
        model.train(False)
        running_vloss = 0.0
        torch.cuda.empty_cache()
        for batch_validation_ct, validation_data in enumerate(validation_loader):
            validation_inputs, BVP_validation_label = validation_data
            vloss, rPPG = validate_batch.val_batch(validation_inputs, BVP_validation_label, model,
                                                                 loss_Inst)
            running_vloss += vloss
            avg_vloss = running_vloss / (batch_validation_ct + 1)
            example_ct_validation += len(validation_inputs)
            print("cuda memory after batch {}".format(batch_ct))
            torch.cuda.memory_summary(device=None, abbreviated=False)
        torch.cuda.empty_cache()
        wandb.log({"epoch": epoch, "loss": avg_vloss})
        print('LOSS train {} valid {}'.format(last_loss, avg_vloss))

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
