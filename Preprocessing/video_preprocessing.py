import os
from fnmatch import fnmatch
import shutil
import numpy as np
# internal modules
from Preprocessing import FaceDetection


# %%  Delete dataset folder (inside of output folder)
def video_pre_ubfc(config):
    noFaceListAllVideos = []
    delete_videos = []
    if os.path.exists(config['dataset_path']) and os.path.isdir(config['dataset_path']):
        shutil.rmtree(config['dataset_path'])
    os.mkdir(config['dataset_path'])


    if os.path.exists(config['tempPathNofile']) and os.path.isdir(config['tempPathNofile']):
        shutil.rmtree(config['tempPathNofile'])
    os.mkdir(config['tempPathNofile'])

    # %% Face detection and save images in folder of video name
    for path, subdirs, files in os.walk(config['gen_path_data']):
        for name in files:
            currentPath = os.path.join(path, name)
            destinationPath = os.path.join(config['dataset_path'], name)
            tempPath = os.path.join(config['tempPathNofile'], name)
            if fnmatch(name, config['pattern_video']):
                os.mkdir(destinationPath)  # make as many folders as videos excist, Foldername is equal the videoname
                os.makedirs(config['tempPathNofile'], exist_ok=True)
                noFaceList = FaceDetection.viola_jonas_face_detector(currentPath, destinationPath, tempPath,
                                                                     config['NEW_FPS_VIDEO'], config['NEW_SIZE_IMAGE'],
                                                                     config)
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
