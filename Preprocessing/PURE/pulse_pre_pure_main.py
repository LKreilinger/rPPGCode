import os
import numpy as np
import shutil
# internal modules
from Preprocessing import pulse_adjust_fps_no_face
from Preprocessing.PURE import pulse_from_json

def pulse_pure(config, noFaceListAllVideos, delete_videos, data_split):
    if os.path.exists(config['tempPathNofile']) and os.path.isdir(config['tempPathNofile']):
        shutil.rmtree(config['tempPathNofile'])
    os.mkdir(config['tempPathNofile'])
    path_raw_data_split = os.path.join(config['genPathData'], data_split)

    if os.path.exists(config['tempPathNofile']) and os.path.isdir(config['tempPathNofile']):
        shutil.rmtree(config['tempPathNofile'])
    os.mkdir(config['tempPathNofile'])
    file_extension = "False"
    for path, subdirs, files in os.walk(path_raw_data_split):
        if files != []:
            test_file = files[0]
            file_extension = os.path.splitext(test_file)[1]
        if path_raw_data_split != path and file_extension == config['patternPuls']:
            for file in files:
                folder_tx_images = file[-7:-5]
                Tx = "T" + folder_tx_images
                Sx = os.path.basename(path)[-2:]
                name = "bvp_" + Sx + "_" + Tx + ".csv"
                currentPath = os.path.join(path, file)
                tempPathName = os.path.join(config['tempPathNofile'], name)
                nameNoExten = os.path.splitext(name)[0]
                tempvidFile = nameNoExten.replace("bvp", "vid")
                correspondingVidName = np.array([tempvidFile])
                correspondingVidNameType = tempvidFile + ".avi"
                if not (correspondingVidNameType in delete_videos):
                    index = noFaceListAllVideos.index(correspondingVidName)
                    noFaceList = noFaceListAllVideos[index + 1]
                    pulse_from_json.pulse_pure(currentPath, tempPathName)
                    pulse_adjust_fps_no_face.pulse_prepro(tempPathName, tempPathName, config["samplingRatePulse"],
                                                          config["newSamplingRatePulse"], noFaceList)

