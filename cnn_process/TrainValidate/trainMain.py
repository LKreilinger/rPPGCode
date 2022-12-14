# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:15:28 2022

@author: Laurens Kreilinger
"""
import os
import torch
from datetime import datetime
import numpy as np
import wandb
import math
# local Packages
from cnn_process.TrainValidate import train_batch, get_pulse, validate_batch
from cnn_process.TestModel import append_matrix, performance_metrics


def train_and_validate_model(model, train_loader, validation_loader, loss_Inst, optimizer, config):
    wandb.watch(model, loss_Inst, log="all", log_freq=10)
    # define our custom x axis metric
    wandb.define_metric("epoch")
    # print(torch.cuda.memory_summary(device=config.device, abbreviated=False))
    # %% train and validate model
    epoch_number = 0
    example_ct = 0.  # number of examples seen
    example_ct_validation = 0
    # saving memory
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
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

            # if math.isnan(loss):
            #     print("loss = NaN")
            #     break

            running_loss += loss.item()


            # Gather data and report
            example_ct += len(inputs)
            if batch_ct % 10 == 9:
                # print(torch.cuda.memory_summary(device=config.device, abbreviated=False))
                last_loss = running_loss / 10

                print(f"Loss after " + str(batch_ct + 1).zfill(4) + f" batches: {last_loss:.3f}")
                running_loss = 0.
        wandb.log({"train_loss": last_loss, "epoch": epoch})
        # if math.isnan(loss):
        #     print("loss = NaN")
        #     break
        # Validate model
        model.eval()
        running_vloss = 0.0
        torch.cuda.empty_cache()
        first_run = 0
        BVP_label_all = np.empty([])
        rPPG_all = np.empty([])
        with torch.no_grad():
            for batch_validation_ct, validation_data in enumerate(validation_loader):
                validation_inputs, BVP_validation_label = validation_data
                vloss, rPPG, BVP_label = validate_batch.val_batch(validation_inputs, BVP_validation_label, model,
                                                                  loss_Inst)
                running_vloss += vloss.item()
                avg_vloss = running_vloss / (batch_validation_ct + 1)
                example_ct_validation += len(validation_inputs)
                torch.cuda.empty_cache()

                rPPG_all, BVP_label_all, first_run = append_matrix.append_truth_prediction_label(
                    BVP_label, rPPG, first_run, rPPG_all, BVP_label_all)
        wandb.log({"val_loss": avg_vloss, "epoch": epoch})

        print(f"Loss train: {last_loss:.3f}" + f" Loss validation: {avg_vloss:.3f}")
        # Calculate performace of model with test data
        MAE, RMSE, R = performance_metrics.eval_model_fft(BVP_label_all, rPPG_all, config)
        wandb.log({"FFT MAE": MAE, "FFT MSE": RMSE, "FFT R": R})
        print(f"FFT Validation MAE: {MAE:.3f}" + f" Validation RMSE: {RMSE:.3f}" + f" Validation R: {R:.3f}")
        try:
            MAE, RMSE, STD = performance_metrics.eval_model(BVP_label_all, rPPG_all, config)
            wandb.log({"MAE": MAE, "MSE": RMSE})
            print(f"Validation MAE: {MAE:.3f}" + f" Validation RMSE: {RMSE:.3f}")
        except Exception:
            print("Could not determine pulse for given signal.")

        epoch_number += 1

    # save last model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = "model_{}".format(timestamp)
    model_path = os.path.join(config.path_model, model_name)
    torch.save(model.state_dict(), model_path)
    print('Finished Training')
