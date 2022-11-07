import numpy as np
import pickle
# local Packages
from cnn_process.TestModel import performance_metrics, append_matrix
from cnn_process.load.load_main import PhysNet
from old_py_files import get_pulse
fps=30

path_Label = r"C:\Users\Chaputa\Documents\Trier\Master\Masterarbeit\output\noFaceList\1\BVP_labelWCD_all.pkl"
path_predict = r"C:\Users\Chaputa\Documents\Trier\Master\Masterarbeit\output\noFaceList\1\rPPG_predictWCD_all.pkl"
open_file = open(path_Label, "rb")
ground_truth = pickle.load(open_file)
open_file.close()

open_file = open(path_predict, "rb")
predicted_label = pickle.load(open_file)
open_file.close()
# get pulse
pulse_label = np.zeros((1, ground_truth.shape[1]))
pulse_predic = np.zeros((1, ground_truth.shape[1]))
for col in range(ground_truth.shape[1]):
    pulse_label[0, col] = get_pulse.get_rfft_pulse(ground_truth[:, col], fps)
    pulse_predic[0, col] = get_pulse.get_rfft_pulse(predicted_label[:, col], fps)

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
    nummerator += (pulse_predic[0, i] - predict_mean) * (pulse_label[0, i] - label_mean)
    denominator1 += (pulse_predic[0, i] - predict_mean) ** 2
    denominator2 += (pulse_label[0, i] - label_mean) ** 2
R = (nummerator / (np.sqrt(denominator1) * np.sqrt(denominator2)))