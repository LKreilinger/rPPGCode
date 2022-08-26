# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:15:28 2022

@author: Laurens Kreilinger
"""
import os
import shutil
import pickle
import numpy as np

# internal modules
from Preprocessing.WCD import convertPolarData, image_preprocessing

from Preprocessing import makeTxt, pulse_adjust_fps_no_face, save_load


def preprocessing_wcd_dataset(config) -> None:
    """
    Main Funktion for preprocessing the UBFC_Phys Dataset
    :rtype: None
    :param gen_path:
    """



    if os.path.exists(config['datasetPath']) and os.path.isdir(config['datasetPath']):
        shutil.rmtree(config['datasetPath'])
    os.mkdir(config['datasetPath'])


    # Video Preprocessing saving frames with detected face
    noFaceListAllVideos, delete_videos = image_preprocessing.faceDetectionImg(config)

    # Save and load list with information about detected faces
    save_load.save_variable(config, noFaceListAllVideos, delete_videos, data_split)
    save_load.load_variable(config, data_split)

    # Pulse preprocessing and fit to frames
    pulse_preprocessing_main.pulse_pre_ubfc(config, noFaceListAllVideos, delete_videos, data_split)

    # Make annotation.txt of the images and the pulse data
    path_dataset_split = os.path.join(config['datasetPath'], data_split)
    makeTxt.makeAnnotation(config['tempPathNofile'], path_dataset_split, config['nFramesVideo'])




    # %% 4: Change sampling rate of pulse data to 30Hz,
    #        delete BVP values if no face detected and
    #        convert data
    #        synchronize BVP data with images

    # Generate temp path for saving temp pulse data
    tempPath = os.path.join(gen_path, "rPPGCode", "temp")
    if os.path.exists(tempPath) and os.path.isdir(tempPath):
        shutil.rmtree(tempPath)
    os.mkdir(tempPath)
    polar_data = os.path.join(gen_path, "data", "WCD", "data_Polar")
    camera_data = os.path.join(gen_path, "data", "WCD", "data_Realsense")
    for path, subdirs, files in os.walk(polar_data):
        if polar_data != path and (os.path.split(os.path.split(path)[0])[1] != "data_Polar"):
            Tx = os.path.basename(path)
            Sx = os.path.split(os.path.split(path)[0])[1]
            app_name = "".join(files)
            name = "bvp_" + Sx + "_" + Tx + ".csv"
            currentPath = os.path.join(path, app_name)
            camera_data_path = os.path.join(camera_data, Sx, Tx)
            tempPathName = os.path.join(tempPath, name)

            nameNoExten = os.path.splitext(name)[0]
            tempvidFile = nameNoExten.replace("bvp", "vid")

            correspondingVidName = np.array([tempvidFile])  #  !!!!!!!!! here T and t
            index = noFaceListAllVideos.index(correspondingVidName)
            noFaceList = noFaceListAllVideos[index + 1]
            convertPolarData.convert_polar_data(currentPath, tempPathName, camera_data_path, SAMPLING_RATE_PULSE, NEW_SAMPLING_RATE,
                                                noFaceList)
            pulse_adjust_fps_no_face.pulse_prepro(tempPathName, tempPathName, SAMPLING_RATE_PULSE, NEW_SAMPLING_RATE,
                                            noFaceList)
            a=1

    # %% Generate annotations.txt in dataset
    #   videoname.png 1 17 0
    #   Name; start Frame; end frame; label (pulsedata)
    # Number of Frames per video -> 128
    tempPath = os.path.join(gen_path + "rPPGCode", "temp")
    makeTxt.makeAnnotation(tempPath, datasetpath, nFramesVideo)