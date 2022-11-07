import numpy as np
import heartpy as hp
from old_py_files import get_pulse

def eval_model(ground_truth, predicted_label, config):
    # get pulse
    pulse_label = np.zeros((1, ground_truth.shape[1]))
    pulse_predic = np.zeros((1, ground_truth.shape[1]))
    for col in range(ground_truth.shape[1]):
        working_data_ground_truth, measures_ground_truth = hp.process(ground_truth[:, col], config.fps)
        pulse_label[0, col] = measures_ground_truth['bpm']
        working_data_predicted_label, measures_predicted_label = hp.process(predicted_label[:, col], config.fps)
        pulse_predic[0, col] = measures_predicted_label['bpm']


    # mean square error (RMSE)
    sum = 0
    for i in range(ground_truth.shape[1]):
        sum += abs(pulse_label[0, i] - pulse_predic[0, i]) ** 2

    RMSE = np.sqrt(sum / ground_truth.shape[1])
    # mean absolute error (MAE)
    sum = 0
    for i in range(ground_truth.shape[1]):
        sum += abs(pulse_label[0, i] - pulse_predic[0, i])

    MAE = sum / ground_truth.shape[1]

    #  standard deviation (SD)
    diff_vector = np.zeros(ground_truth.shape[1])
    for i in range(ground_truth.shape[1]):
        diff_vector[i] = abs(pulse_label[0, i] - pulse_predic[0, i])
    STD = np.std(diff_vector)


    return MAE, RMSE, STD

def eval_model_fft(ground_truth, predicted_label, config):
    # get pulse
    pulse_label = np.zeros((1, ground_truth.shape[1]))
    pulse_predic = np.zeros((1, ground_truth.shape[1]))
    for col in range(ground_truth.shape[1]):
        pulse_label[0, col] = get_pulse.get_rfft_pulse(ground_truth[:, col], config.fps)
        pulse_predic[0, col] = get_pulse.get_rfft_pulse(predicted_label[:, col], config.fps)

    # mean square error (RMSE)
    sum = 0
    for i in range(ground_truth.shape[1]):
        sum += (pulse_predic[0, i] - pulse_label[0, i]) ** 2

    RMSE = np.sqrt(sum / ground_truth.shape[1])
    # mean absolute error (MAE)
    sum = 0
    for i in range(ground_truth.shape[1]):
        sum += abs(pulse_predic[0, i] - pulse_label[0, i])

    MAE = sum / ground_truth.shape[1]

    # #  standard deviation (SD)
    # diff_vector = np.zeros(ground_truth.shape[1])
    # for i in range(ground_truth.shape[1]):
    #     diff_vector[i] = abs(pulse_label[0, i] - pulse_predic[0, i])
    # STD = np.std(diff_vector)

    # pearson correlation coefficien (R)
    nummerator = 0
    denominator1 = 0
    denominator2 = 0
    label_mean = np.mean(pulse_label)
    predict_mean = np.mean(pulse_predic)
    for i in range(ground_truth.shape[1]):
        nummerator += (pulse_predic[0, i] - predict_mean)*(pulse_label[0, i] - label_mean)
        denominator1 += (pulse_predic[0, i] - predict_mean)
        denominator2 += (pulse_label[0, i] - label_mean)
    R = (nummerator / (np.sqrt(denominator1)*np.sqrt(denominator2)))

    return MAE, RMSE, R
