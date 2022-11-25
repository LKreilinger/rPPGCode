import numpy as np
import pickle
import matplotlib.pyplot as plt
import heartpy as hp
# local Packages
from cnn_process.TestModel import performance_metrics, append_matrix
from cnn_process.load.load_main import PhysNet
from old_py_files import get_pulse
fps=30

path_Label = r"C:\Users\Chaputa\Documents\Trier\Master\Masterarbeit\output\noFaceList\1\BVP_labelWCD_s2_t3.pkl"
path_predict = r"C:\Users\Chaputa\Documents\Trier\Master\Masterarbeit\output\noFaceList\3\BVP_labelWCD.pkl"
open_file = open(path_Label, "rb")
ground_truth = pickle.load(open_file)
open_file.close()

open_file = open(path_predict, "rb")
predicted_label = pickle.load(open_file)
open_file.close()
# get pulse
pulse_label = np.zeros((1, ground_truth.shape[1]))
pulse_predic = np.zeros((1, ground_truth.shape[1]))
# np.savetxt("ground_truth_PURE_Split_1.txt", ground_truth[:,0])
# np.savetxt("predicted_label_PURE_Split_1.txt", predicted_label[:,0])
for col in range(ground_truth.shape[1]):
    pulse_label[0, col] = get_pulse.get_rfft_pulse(ground_truth[:, col], fps)
#    pulse_predic[0, col] = get_pulse.get_rfft_pulse(predicted_label[:, col], fps)
a_min=np.min(pulse_label)
a_max=np.max(pulse_label)
a_mean=np.mean(pulse_label)
a_std=np.std(pulse_label)

# get pulse
pulse_label2 = np.zeros((1, ground_truth.shape[1]))
pulse_predic2 = np.zeros((1, ground_truth.shape[1]))
for col in range(ground_truth.shape[1]):
    working_data_ground_truth, measures_ground_truth = hp.process(ground_truth[:, col], fps)
    pulse_label2[0, col] = measures_ground_truth['bpm']
    #working_data_predicted_label, measures_predicted_label = hp.process(predicted_label[:, col], fps)
    #pulse_predic2[0, col] = measures_predicted_label['bpm']
a_min2=np.min(pulse_label2)
a_max2=np.max(pulse_label2)
a_mean2=np.mean(pulse_label2)
a_std2=np.std(pulse_label2)

bins = [40, 45,50,55, 60,65, 70,75, 80,85, 90,95, 100,105, 110,115, 120,125, 130,135, 140,145, 150,155, 160,165, 170,175, 180]
hist, bins = np.histogram(pulse_label, bins=bins)
width = np.diff(bins)
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist)
plt.show()
# # mean square error (RMSE)
# sum = 0
# for i in range(ground_truth.shape[1]):
#     sum += (pulse_predic[0, i] - pulse_label[0, i]) ** 2
#
# RMSE = np.sqrt(sum / ground_truth.shape[1])
# # mean absolute error (MAE)
# sum = 0
# for i in range(ground_truth.shape[1]):
#     sum += abs(pulse_predic[0, i] - pulse_label[0, i])
#
# MAE = sum / ground_truth.shape[1]
#
# # #  standard deviation (SD)
# # diff_vector = np.zeros(ground_truth.shape[1])
# # for i in range(ground_truth.shape[1]):
# #     diff_vector[i] = abs(pulse_label[0, i] - pulse_predic[0, i])
# # STD = np.std(diff_vector)
#
# # pearson correlation coefficien (R)
# nummerator = 0
# denominator1 = 0
# denominator2 = 0
# label_mean = np.mean(pulse_label)
# predict_mean = np.mean(pulse_predic)
# for i in range(ground_truth.shape[1]):
#     nummerator += (pulse_predic[0, i] - predict_mean) * (pulse_label[0, i] - label_mean)
#     denominator1 += (pulse_predic[0, i] - predict_mean) ** 2
#     denominator2 += (pulse_label[0, i] - label_mean) ** 2
# R = (nummerator / (np.sqrt(denominator1) * np.sqrt(denominator2)))