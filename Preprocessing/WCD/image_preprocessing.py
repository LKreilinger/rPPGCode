import os
import numpy as np
import shutil
# internal modules
from Preprocessing.WCD import faceDetectionImg


def image_pre_WCD(config):
    noFaceListAllVideos = []
    delete_videos = []
    for path, subdirs, files in os.walk(config['dataImages']):
        if config['dataImages'] != path and (os.path.split(os.path.split(path)[0])[1] != "data_Realsense"):
            Tx = os.path.basename(path)
            Sx = os.path.split(os.path.split(path)[0])[1]
            name = "vid_" + Sx + "_" + Tx + ".avi"
            destinationPath = os.path.join(config['datasetPath'], name)
            if not (os.path.exists(destinationPath) and os.path.isdir(destinationPath)):
                os.mkdir(destinationPath)
            noFaceList = faceDetectionImg.viola_jonas_face_detector_img(path, destinationPath,
                                                                        NEW_SIZE_IMAGE)
            nameNoExten = os.path.splitext(name)[0]
            noFaceListAllVideos.append(nameNoExten)
            noFaceListAllVideos.append(noFaceList)
            n_zeros = np.count_nonzero(noFaceList == 0)
            n_ones = np.count_nonzero(noFaceList == 1)
            shutil.rmtree(config['tempPathNofile'])
            if n_zeros > n_ones:
                nameNoExten = os.path.splitext(name)[0]
                noFaceListAllVideos.append(nameNoExten)
                noFaceListAllVideos.append(noFaceList)
            else:
                shutil.rmtree(destinationPath)
                delete_videos.append(name)

    return noFaceListAllVideos, delete_videos
