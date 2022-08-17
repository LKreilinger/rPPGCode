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
    if os.path.exists(config['datasetPath']) and os.path.isdir(config['datasetPath']):
        shutil.rmtree(config['datasetPath'])
    os.mkdir(config['datasetPath'])


    if os.path.exists(config['tempPathNofile']) and os.path.isdir(config['tempPathNofile']):
        shutil.rmtree(config['tempPathNofile'])
    os.mkdir(config['tempPathNofile'])

    # %% Face detection and save images in folder of video name
    for path, subdirs, files in os.walk(config['genPathData']):
        for name in files:
            currentPath = os.path.join(path, name)
            destinationPath = os.path.join(config['datasetPath'], name)
            tempPath = os.path.join(config['tempPathNofile'], name)
            if fnmatch(name, config['patternVideo']):
                os.mkdir(destinationPath)  # make as many folders as videos excist, Foldername is equal the videoname
                os.makedirs(config['tempPathNofile'], exist_ok=True)
                noFaceList = FaceDetection.viola_jonas_face_detector(currentPath, destinationPath, tempPath,
                                                                     config['newFpsVideo'], config['newSizeImage'],
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
