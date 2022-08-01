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
from Preprocessing.WCD import convertPolarData, faceDetectionImg
from Preprocessing import makeTxt, pulsePreprocessing


def preprocessing_wcd_dataset(gen_path: str, nFramesVideo) -> None:
    """
    Main Funktion for preprocessing the UBFC Dataset
    :rtype: None
    :param gen_path:
    """

    # %% Preprocessing Video and Pulse data from the folder Data
    # 1: Viola Face Detector
    # 2: Format of face videos: FPS=30; sice=128x128; length a multiple of 128 frames
    # 3:  save every Frame as img_00001.jpg, img_00002.jpg...
    SAMPLING_RATE_PULSE: int = 55  # in Hz of pulse data
    NEW_SAMPLING_RATE: int = 30  # for pulse data and video
    NEW_SIZE_IMAGE = (128, 128)  # of the face
    noFaceListAllVideos = []
    # %%  Delete dataset folder (inside of output folder)
    datasetpath = os.path.join(gen_path, "output", "WCDDataset")
    camera_data = os.path.join(gen_path, "data", "WCD", "data_Realsense")
    if os.path.exists(datasetpath) and os.path.isdir(datasetpath):
        shutil.rmtree(datasetpath)
    os.mkdir(datasetpath)
    for path, subdirs, files in os.walk(camera_data):
        if camera_data != path and (os.path.split(os.path.split(path)[0])[1] != "data_Realsense"):
            Tx = os.path.basename(path)
            Sx = os.path.split(os.path.split(path)[0])[1]
            name = "vid_" + Sx + "_" + Tx + ".avi"
            destinationPath = os.path.join(datasetpath, name)
            if not (os.path.exists(destinationPath) and os.path.isdir(destinationPath)):
                os.mkdir(destinationPath)
            noFaceList = faceDetectionImg.viola_jonas_face_detector_img(path, destinationPath,
                                                                        NEW_SIZE_IMAGE)
            nameNoExten = os.path.splitext(name)[0]
            noFaceListAllVideos.append(nameNoExten)
            noFaceListAllVideos.append(noFaceList)

    #%%save noFaceListAllVideos
    file_name = r"/code/Preprocessing/WCD/noFaceListAllVideos.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(noFaceListAllVideos, open_file)
    open_file.close()

    # open noFaceListAllVideos
    file_name = r"/code/Preprocessing/WCD/noFaceListAllVideos.pkl"
    open_file = open(file_name, "rb")
    noFaceListAllVideos = pickle.load(open_file)
    open_file.close()

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
            pulsePreprocessing.pulse_prepro(tempPathName, tempPathName, SAMPLING_RATE_PULSE, NEW_SAMPLING_RATE,
                                            noFaceList)
            a=1

    # %% Generate annotations.txt in dataset
    #   videoname.png 1 17 0
    #   Name; start Frame; end frame; label (pulsedata)
    # Number of Frames per video -> 128
    tempPath = os.path.join(gen_path + "rPPGCode", "temp")
    makeTxt.makeAnnotation(tempPath, datasetpath, nFramesVideo)