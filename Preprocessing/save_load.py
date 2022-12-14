import os
import shutil
import pickle

def save_variable(config, noFaceListAllVideos, delete_videos, data_split):
    # save noFaceListAllVideos and delete_videos
    name_no_face = "noFaceListAllVideos_" + data_split + ".pkl"
    name_deleted_videos = "delete_videos_" + data_split + ".pkl"
    list_path = config['variblesPath']
    path_deleted_videos = os.path.join(list_path, name_deleted_videos)
    path_no_face = os.path.join(list_path, name_no_face)

    if os.path.exists(list_path) and os.path.isdir(list_path):
        shutil.rmtree(list_path)
    os.mkdir(list_path)

    open_file = open(path_no_face, "wb")
    pickle.dump(noFaceListAllVideos, open_file)
    open_file.close()

    open_file = open(path_deleted_videos, "wb")
    pickle.dump(delete_videos, open_file)
    open_file.close()

def load_variable(config, data_split):
    # open noFaceListAllVideos and delete_videos
    name_no_face = "noFaceListAllVideos_" + data_split + ".pkl"
    name_deleted_videos = "delete_videos_" + data_split + ".pkl"
    list_path = config['variblesPath']
    path_deleted_videos = os.path.join(list_path, name_deleted_videos)
    path_no_face = os.path.join(list_path, name_no_face)

    open_file = open(path_no_face, "rb")
    noFaceListAllVideos = pickle.load(open_file)
    open_file.close()

    open_file = open(path_deleted_videos, "rb")
    delete_videos = pickle.load(open_file)
    open_file.close()
    return noFaceListAllVideos, delete_videos
