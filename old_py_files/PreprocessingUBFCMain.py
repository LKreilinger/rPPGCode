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
from Preprocessing import makeTxt, pulse_adjust_fps_no_face, FaceDetection


def preprocessing_ubfc_dataset(gen_path: str, nFramesVideo, workingPath, docker) -> None:
    """
    Main Funktion for preprocessing the UBFC_Phys Dataset
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
    delete_videos = []
    patternVideo = "*.avi"
    patternPuls = "*.csv"

    # %%  Delete dataset folder (inside of output folder)

    datasetpath = os.path.join(gen_path, "output", "UBFCDataset")
    gen_path_data = os.path.join(gen_path, "data", "UBFC_Phys")
    if os.path.exists(datasetpath) and os.path.isdir(datasetpath):
        shutil.rmtree(datasetpath)
    os.mkdir(datasetpath)

    # Generate temp path for video and BCP-data processing
    if docker:
        tempPathNofile = os.path.join(workingPath, "output", "temp")
    else:
        tempPathNofile = os.path.join(workingPath, "temp")
    if os.path.exists(tempPathNofile) and os.path.isdir(tempPathNofile):
        shutil.rmtree(tempPathNofile)
    os.mkdir(tempPathNofile)

    #%% Face detection and save images in folder of video name
    for path, subdirs, files in os.walk(gen_path_data):
        for name in files:
            currentPath = os.path.join(path, name)
            destinationPath = os.path.join(datasetpath, name)
            tempPath = os.path.join(tempPathNofile, name)
            if fnmatch(name, patternVideo):
                os.mkdir(destinationPath)  # make as many folders as videos excist, Foldername is equal the videoname
                os.makedirs(tempPathNofile, exist_ok=True)
                noFaceList = FaceDetection.viola_jonas_face_detector(currentPath, destinationPath, tempPath,
                                                                     NEW_SAMPLING_RATE, NEW_SIZE_IMAGE)
                n_zeros = np.count_nonzero(noFaceList == 0)
                n_ones = np.count_nonzero(noFaceList == 1)
                shutil.rmtree(tempPathNofile)
                if n_zeros > n_ones:
                    nameNoExten = os.path.splitext(name)[0]
                    noFaceListAllVideos.append(nameNoExten)
                    noFaceListAllVideos.append(noFaceList)
                else:
                    shutil.rmtree(destinationPath)
                    delete_videos.append(name)
    #%%
    # save noFaceListAllVideos and delete_videos
    name_no_face = "noFaceListAllVideos.pkl"
    name_deleted_videos = "delete_videos.pkl"
    list_path = os.path.join(gen_path, "output", "noFaceList")
    path_deleted_videos = os.path.join(list_path, name_deleted_videos)
    path_no_face = os.path.join(list_path, name_no_face)

    if os.path.exists(list_path) and os.path.isdir(list_path):
        shutil.rmtree(list_path)
    os.mkdir(list_path)

    open_file = open(path_no_face, "wb")
    pickle.dump(noFaceListAllVideos, open_file)
    open_file.close()

    open_file = open(path_deleted_videos, "wb")
    pickle.dump(delete_videos, open_file)
    open_file.close()

    # open noFaceListAllVideos and delete_videos
    open_file = open(path_no_face, "rb")
    noFaceListAllVideos = pickle.load(open_file)
    open_file.close()

    open_file = open(path_deleted_videos, "rb")
    delete_videos = pickle.load(open_file)
    open_file.close()

    # %% 4: Change sampling rate of pulse data to 30Hz and delet BVP values if no face detected

    if os.path.exists(tempPathNofile) and os.path.isdir(tempPathNofile):
        shutil.rmtree(tempPathNofile)
    os.mkdir(tempPathNofile)
    for path, subdirs, files in os.walk(gen_path_data):
        for name in files:
            currentPath = os.path.join(path, name)
            tempPath = os.path.join(tempPathNofile, name)
            if fnmatch(name, patternPuls) and name[0] == 'b':
                nameNoExten = os.path.splitext(name)[0]
                tempvidFile = nameNoExten.replace("bvp", "vid")
                correspondingVidName = np.array([tempvidFile])
                correspondingVidNameType = tempvidFile + ".avi"
                if not(correspondingVidNameType in delete_videos):
                    index = noFaceListAllVideos.index(correspondingVidName)
                    noFaceList = noFaceListAllVideos[index + 1]
                    pulse_data_change_format.pulse_prepro(currentPath, tempPath, SAMPLING_RATE_PULSE, NEW_SAMPLING_RATE,
                                                noFaceList)

    # %% Generate annotations.txt in dataset
    #   videoname.png 1 17 0
    #   Name; start Frame; end frame; label (pulsdata)
    # Number of Frames per video -> 128
    makeTxt.makeAnnotation(tempPathNofile, datasetpath, nFramesVideo)
