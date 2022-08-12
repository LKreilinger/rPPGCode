import os
import pickle
import numpy as np

list_path = r"C:\Users\Chaputa\Documents\Trier\Master\Masterarbeit\output\noFaceList"
#open
list_name = "noFaceListAllVideos.pkl"
file_path_name = os.path.join(list_path, list_name)
# open noFaceListAllVideos
open_file = open(file_path_name, "rb")
noFaceListAllVideos = pickle.load(open_file)
open_file.close()

list_name = "delete_videos.pkl"
file_path_name = os.path.join(list_path, list_name)
open_file = open(file_path_name, "rb")
deleted_videos = pickle.load(open_file)
open_file.close()
name = np.array(["vid_s1_T1"])
index = noFaceListAllVideos.index(np.array([name]))
# save
# noFaceListAllVideos[10]="vid_s40_T3"
open_file = open(file_path_name, "wb")
pickle.dump(noFaceListAllVideos, open_file)
open_file.close()