import numpy as np


def append_truth_prediction_label(ground_truth, predicted_label, first_run, predicted_label_all, ground_truth_all):
    predicted_label_np = predicted_label.detach().numpy()
    ground_truth_np = ground_truth.detach().numpy()
    if first_run == 0:
        predicted_label_all = predicted_label_np
        ground_truth_all = ground_truth_np
        first_run = 1
    else:
        predicted_label_all = np.concatenate((predicted_label_all, predicted_label_np), axis=1)
        ground_truth_all = np.concatenate((ground_truth_all, ground_truth_np), axis=1)


    return predicted_label_all, ground_truth_all, first_run
