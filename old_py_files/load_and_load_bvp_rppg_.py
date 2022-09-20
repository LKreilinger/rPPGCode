import os
import pickle
import numpy as np
from old_py_files import get_pulse

list_path = r"C:\Users\Chaputa\Documents\Trier\Master\Masterarbeit\output\noFaceList"
#open BVP_label_all
list_name = "BVP_label_all.pkl"
file_path_name = os.path.join(list_path, list_name)
open_file = open(file_path_name, "rb")
BVP_label_all = pickle.load(open_file)
open_file.close()

#open rPPG_all
list_name = "rPPG_all.pkl"
file_path_name = os.path.join(list_path, list_name)
# open noFaceListAllVideos
open_file = open(file_path_name, "rb")
rPPG_all = pickle.load(open_file)
open_file.close()

fps = 30
# get pulse
pulse_label = np.zeros((1, BVP_label_all.shape[1]))
pulse_predic = np.zeros((1, BVP_label_all.shape[1]))
for col in range(BVP_label_all.shape[1]):
    pulse_label[0, col] = get_pulse.get_rfft_pulse(BVP_label_all[:, col], fps)  # get pulse from signal
    pulse_predic[0, col] = get_pulse.get_rfft_pulse(rPPG_all[:, col], fps)


# mean square error (RMSE)
sum = 0
for i in range(BVP_label_all.shape[1]):
    sum += abs(pulse_label[0, i] - pulse_predic[0, i]) ** 2

RMSE = np.sqrt(sum / BVP_label_all.shape[1])
# mean absolute error (MAE)
sum = 0
for i in range(BVP_label_all.shape[1]):
    sum += abs(pulse_label[0, i] - pulse_predic[0, i])

MAE = sum / BVP_label_all.shape[1]

#  standard deviation (SD)
diff_vector = np.zeros(BVP_label_all.shape[1])
for i in range(BVP_label_all.shape[1]):
    diff_vector[i] = abs(pulse_label[0, i] - pulse_predic[0, i])
STD = np.std(diff_vector)


