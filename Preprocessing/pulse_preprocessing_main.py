import os
import shutil
from fnmatch import fnmatch
import numpy as np
# internal modules
from Preprocessing import pulse_adjust_fps_no_face
from Preprocessing.UBFC_rPPG import pulse_ubfc_rppg_convert


def pulse_pre_ubfc(config, noFaceListAllVideos, delete_videos, data_split):
    if os.path.exists(config['tempPathNofile']) and os.path.isdir(config['tempPathNofile']):
        shutil.rmtree(config['tempPathNofile'])
    os.mkdir(config['tempPathNofile'])
    path_raw_data_split = os.path.join(config['genPathData'], data_split)
    for path, subdirs, files in os.walk(path_raw_data_split):
        for name in files:
            currentPath = os.path.join(path, name)
            tempPath = os.path.join(config['tempPathNofile'], name)
            if fnmatch(name, config['patternPuls']) and name[0] == 'b':
                nameNoExten = os.path.splitext(name)[0]
                tempvidFile = nameNoExten.replace("bvp", "vid")
                correspondingVidName = np.array([tempvidFile])
                correspondingVidNameType = tempvidFile + ".avi"
                if not(correspondingVidNameType in delete_videos):
                    index = noFaceListAllVideos.index(correspondingVidName)
                    noFaceList = noFaceListAllVideos[index + 1]
                    if os.path.splitext(name)[-1] == ".txt":
                        tempPath = os.path.join(config['tempPathNofile'], nameNoExten + ".csv")
                        pulse_ubfc_rppg_convert.pulse_convert(currentPath, tempPath)
                        pulse_adjust_fps_no_face.pulse_prepro(tempPath, tempPath,
                                                              config['samplingRatePulse'],
                                                              config['newSamplingRatePulse'],
                                                              noFaceList)
                    else:
                        pulse_adjust_fps_no_face.pulse_prepro(currentPath, tempPath,
                                                              config['samplingRatePulse'], config['newSamplingRatePulse'],
                                                              noFaceList)
