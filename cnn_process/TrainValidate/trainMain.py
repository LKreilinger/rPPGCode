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
    #print(torch.cuda.memory_summary(device=config.device, abbreviated=False))
    # %% train and validate model
    epoch_number = 0.
    example_ct = 0.  # number of examples seen
    example_ct_validation = 0
    # saving memory
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
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
            # Gather data and report
            example_ct += len(inputs)
            if batch_ct % 10 == 9:
                #print(torch.cuda.memory_summary(device=config.device, abbreviated=False))
                last_loss = running_loss / 10
                wandb.log({"epoch": epoch, "train_loss": last_loss})
                print(f"Loss after " + str(batch_ct + 1).zfill(4) + f" batches: {last_loss:.3f}")
                running_loss = 0.

        # Validate model
        model.eval()
        running_vloss = 0.0
        torch.cuda.empty_cache()
        with torch.no_grad():
            for batch_validation_ct, validation_data in enumerate(validation_loader):
                validation_inputs, BVP_validation_label = validation_data
                vloss, rPPG = validate_batch.val_batch(validation_inputs, BVP_validation_label, model,
                                                       loss_Inst)
                running_vloss += vloss.item()
                avg_vloss = running_vloss / (batch_validation_ct + 1)
                example_ct_validation += len(validation_inputs)
                torch.cuda.empty_cache()
                if batch_validation_ct % 10 == 9:
                    wandb.log({"epoch": epoch, "val_loss": avg_vloss})

        # torch.cuda.memory_summary(device=None, abbreviated=False)
        print(f"Loss train: {last_loss:.3f}" + f" Loss validation: {avg_vloss:.3f}")

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
    model_name = "model_{}".format(timestamp)
    model_path = os.path.join(config.path_model, model_name)
    torch.save(model.state_dict(), model_path)
    print('Finished Training')
