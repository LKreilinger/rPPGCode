# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:15:28 2022

@author: Laurens Kreilinger
"""
import os
import shutil
from fnmatch import fnmatch
import numpy as np
import pickle
# internal modules
from Preprocessing.UBFC import FaceDetection
from Preprocessing import makeTxt, pulsePreprocessing


def preprocessing_ubfc_dataset(gen_path: str, nFramesVideo) -> None:
    """
    Main Funktion for preprocessing the UBFC Dataset
    :rtype: None
    :param gen_path:
    """

    #%% Preprocessing Video and Pulse data from the folder Data
    # 1: Viola Face Detector
    # 2: Format of face videos: FPS=30; sice=128x128; length a multiple of 128 frames
    # 3:  save every Frame as img_00001.jpg, img_00002.jpg...
    SAMPLING_RATE_PULSE: int = 64  # in Hz of pulse data
    NEW_SAMPLING_RATE: int = 30  # for pulse data and video
    NEW_SIZE_IMAGE = (128, 128)  # of the face
    noFaceListAllVideos = []
    patternVideo = "*.avi"
    patternPuls = "*.csv"

    # %%  Delete dataset folder (inside of output folder)
    datasetpath = os.path.join(gen_path + '\\output\\UBFCDataset')
    gen_path = os.path.join(gen_path + '\\data\\UBFC')
    if os.path.exists(datasetpath) and os.path.isdir(datasetpath):
        shutil.rmtree(datasetpath)
    os.mkdir(datasetpath)
    for path, subdirs, files in os.walk(gen_path):
        for name in files:
            currentPath = os.path.join(path + '\\' + name)
            destinationPath = os.path.join(datasetpath + '\\' + name)
            # Generate temp path for video processing
            tempPath = currentPath.replace("data", "rPPGCode")
            tempPath = os.path.dirname(os.path.dirname(os.path.dirname(tempPath)))
            tempPath = os.path.join(tempPath, 'temp\\' + name)
            tempPathNofile = os.path.dirname(tempPath)

            if fnmatch(name, patternVideo):
                os.mkdir(destinationPath)  # make as many folders as videos excist, Foldername is equal the videoname
                os.makedirs(tempPathNofile, exist_ok=True)
                noFaceList = FaceDetection.viola_jonas_face_detector(currentPath, destinationPath, tempPath,
                                                                     NEW_SAMPLING_RATE, NEW_SIZE_IMAGE)
                shutil.rmtree(tempPathNofile)
                nameNoExten = os.path.splitext(name)[0]
                noFaceListAllVideos.append(nameNoExten)
                noFaceListAllVideos.append(noFaceList)
    #%%
    # save noFaceListAllVideos
    file_name = r'/code/Preprocessing\UBFC\UBFC\noFaceListAllVideos.pkl'
    open_file = open(file_name, "wb")
    pickle.dump(noFaceListAllVideos, open_file)
    open_file.close()

    # open noFaceListAllVideos
    file_name = r'/code/Preprocessing/UBFC/noFaceListAllVideos.pkl'
    open_file = open(file_name, "rb")
    noFaceListAllVideos = pickle.load(open_file)
    open_file.close()

    # %% 4: Change sampling rate of pulse data to 30Hz and delet BVP values if no face detected
    for path, subdirs, files in os.walk(gen_path):
        for name in files:
            currentPath = os.path.join(path + '\\' + name)
            # Generate temp path for saving temp pulse data
            tempPath = currentPath.replace("data", "rPPGCode")
            tempPath = os.path.dirname(os.path.dirname(os.path.dirname(tempPath)))
            tempPath = os.path.join(tempPath, 'temp\\' + name)
            if fnmatch(name, patternPuls):
                nameNoExten = os.path.splitext(name)[0]
                tempvidFile = nameNoExten.replace("bvp", "vid")
                correspondingVidName = np.array([tempvidFile])

                index = noFaceListAllVideos.index(correspondingVidName)
                noFaceList = noFaceListAllVideos[index + 1]
                pulsePreprocessing.pulse_prepro(currentPath, tempPath, SAMPLING_RATE_PULSE, NEW_SAMPLING_RATE,
                                                noFaceList)

    # %% Generate annotations.txt in dataset
    #   videoname.png 1 17 0
    #   Name; start Frame; end frame; label (pulsdata)
    # Number of Frames per video -> 128
    tempPath = os.path.dirname(gen_path)
    tempPath = os.path.dirname(tempPath)
    tempPath = os.path.join(tempPath + '\\rPPGCode\\temp')
    makeTxt.makeAnnotation(tempPath, datasetpath, nFramesVideo)
