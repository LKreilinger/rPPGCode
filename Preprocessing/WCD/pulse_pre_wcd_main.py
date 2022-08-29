import os
import numpy as np
import shutil
# internal modules
from Preprocessing.WCD import convertPolarData

from Preprocessing import pulse_adjust_fps_no_face

def pulse_wcd(config, noFaceListAllVideos, delete_videos):
    if os.path.exists(config['tempPathNofile']) and os.path.isdir(config['tempPathNofile']):
        shutil.rmtree(config['tempPathNofile'])
    os.mkdir(config['tempPathNofile'])
    for path, subdirs, files in os.walk(config['dataPulse']):
        if config['dataPulse'] != path and (os.path.split(os.path.split(path)[0])[1] != "data_Polar"):
            for file in files:
                if "PPG" in file:
                    Tx = os.path.basename(path)
                    Sx = os.path.split(os.path.split(path)[0])[1]
                    #app_name = "".join(files)
                    name = "bvp_" + Sx + "_" + Tx + ".csv"
                    currentPath = os.path.join(path, file)
                    camera_data_path = os.path.join(config['dataImages'], Sx, Tx)
                    tempPathName = os.path.join(config['tempPathNofile'], name)
                    nameNoExten = os.path.splitext(name)[0]
                    tempvidFile = nameNoExten.replace("bvp", "vid")
                    correspondingVidName = np.array([tempvidFile])
                    correspondingVidNameType = tempvidFile + ".avi"
                    if not (correspondingVidNameType in delete_videos):
                        index = noFaceListAllVideos.index(correspondingVidName)
                        noFaceList = noFaceListAllVideos[index + 1]
                        convertPolarData.convert_polar_data(currentPath, tempPathName, camera_data_path)
                        pulse_adjust_fps_no_face.pulse_prepro(tempPathName, tempPathName, config["samplingRatePulse"], config["newSamplingRatePulse"],
                                                              noFaceList)
