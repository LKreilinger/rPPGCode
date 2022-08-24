import numpy as np
import heartpy as hp

def eval_model(ground_truth, predicted_label, config):
    # get pulse
    pulse_label = np.zeros((1, ground_truth.shape[1]))
    pulse_predic = np.zeros((1, ground_truth.shape[1]))
    for col in range(ground_truth.shape[1]):
        working_data_ground_truth, measures_ground_truth = hp.process(ground_truth[:, col], config.fps)
        pulse_label[0, col] = measures_ground_truth['bpm']
        working_data_predicted_label, measures_predicted_label = hp.process(predicted_label[:, col], config.fps)
        pulse_predic[0, col] = measures_predicted_label['bpm']


    # mean square error (MSE)
    sum = 0
    for i in range(ground_truth.shape[1]):
        sum += abs(pulse_label[0, i] - pulse_predic[0, i]) ** 2

    MSE = sum / ground_truth.shape[1]
    # mean absolute error (MAE)
    sum = 0
    for i in range(ground_truth.shape[1]):
        sum += abs(pulse_label[0, i] - pulse_predic[0, i])

    MAE = sum / ground_truth.shape[1]

    return MAE, MSE
