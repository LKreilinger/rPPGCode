import os
import numpy as np
import shutil
# internal modules
from Preprocessing.WCD import faceDetectionImg


def image_pre_pure(config, data_split):
    path_dataset_split = os.path.join(config['datasetPath'], data_split)
    path_raw_data_split = os.path.join(config['genPathData'], data_split)
    noFaceListAllVideos = []
    delete_videos = []
    if os.path.exists(path_dataset_split) and os.path.isdir(path_dataset_split):
        shutil.rmtree(path_dataset_split)
    os.mkdir(path_dataset_split)
    for path, subdirs, files in os.walk(path_raw_data_split):
        if path_raw_data_split != path and subdirs == []:
            folder_tx_images = os.path.basename(path)[-2:]
            Tx = "T" + folder_tx_images
            Sx = os.path.split(os.path.split(path)[0])[1]
            name = "vid_" + Sx + "_" + Tx + ".avi"
            destinationPath = os.path.join(path_dataset_split, name)
            if not (os.path.exists(destinationPath) and os.path.isdir(destinationPath)):
                os.mkdir(destinationPath)
            noFaceList = faceDetectionImg.viola_jonas_face_detector_img(path, destinationPath,
                                                                        config["newSizeImage"], config)
            n_zeros = np.count_nonzero(noFaceList == 0)
            n_ones = np.count_nonzero(noFaceList == 1)
            if n_zeros > n_ones:
                nameNoExten = os.path.splitext(name)[0]
                noFaceListAllVideos.append(nameNoExten)
                noFaceListAllVideos.append(noFaceList)
            else:
                shutil.rmtree(destinationPath)
                delete_videos.append(name)
                print("Deleted: ", name)

    return noFaceListAllVideos, delete_videos
