import os
import pickle

list_path = r"C:\Users\Chaputa\Documents\Trier\Master\Masterarbeit\output\noFaceList"

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