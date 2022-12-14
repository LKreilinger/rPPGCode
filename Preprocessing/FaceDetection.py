# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:18:27 2022
"""

import cv2
from moviepy.editor import *
import numpy as np
import shutil
import warnings
# internal modules
from Preprocessing.Augmentation import augmentations


def viola_jonas_face_detector(currentPath: str, destinationPath: str, tempPath: str, config, destinationPath_augmen):
    """
    Face detection with Viola Jonas Algorythm.
    Saving every resized (Face)Frame separately
    # Preprocessing video and pulse data from currentPath
    # 1: Viola Face Detector
    # 2: Format of face videos: FPS=30; sice=128x128; length a multiple of 128 frames
    # 3: Change sampling rate of pulse data to 30Hz
    # 4: Save Frames in destinationPath
    :return: noFaceList
    :rtype: np.ndarray
    :param currentPath:
    :param destinationPath:
    :param tempPath:
    """
    # load trained modul for face detection
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    # Change fps to NewSamplingRate
    clip = VideoFileClip(currentPath)
    fpsOriginal = clip.fps
    # !!!!! only first n seconds!!!!
    # clip = clip.subclip(0, 5)
    # !!!!! only first n seconds!!!!
    # %%
    if fpsOriginal < config['newFpsVideo'] + 1:
        # copy file to tempPath
        shutil.copyfile(currentPath, tempPath)
        # !!!!! only first n seconds!!!!
        # clip = VideoFileClip(currentPath)
        # clip = clip.subclip(0, 5)
        # clip.write_videofile(tempPath, fps=config['newFpsVideo'], codec="libx264")
        # !!!!! only first n seconds!!!!
    else:
        clip.write_videofile(tempPath, fps=config['newFpsVideo'], codec="libx264")

    video_capture = cv2.VideoCapture(tempPath)
    FRAME_COUNT = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    noFaceList = np.zeros(FRAME_COUNT)  # 0=face detectet; 1=no face detected
    base_string = "img"
    firstRectagle = 0  # selecting first rectangele size for all, in one video
    iteratImagIndex = 0

    for iteratingFrames in range(FRAME_COUNT):
        ret, frame = video_capture.read()
        # # only to show single faceROI
        # from matplotlib import pyplot as plt
        # plt.imshow(frame, interpolation='nearest')
        # plt.show()


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=config['scaleFactor'],
                                             minNeighbors=config['minNeighbors'],
                                             minSize=config['minSize'],
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        # define size of new video
        if firstRectagle == 0 and len(faces) != 0:
            (xF, yF, wF, hF) = faces.T
            if wF.__len__() == 1:
                firstRectagle = 1
                wF = int(wF)
                hF = int(hF)
            else:
                warnings.warn('Warning Message: Face detection detected more than one face')
                noFaceList[iteratingFrames] = 1

        # write video frame by frame
        if len(faces) != 0:
            (x, y, w, h) = faces.T
            if x.__len__() == 1:
                x = int(x)
                y = int(y)
                faceROI = frame[y:y + hF, x:x + wF]
                # # only to show single faceROI
                # from matplotlib import pyplot as plt
                # plt.imshow(faceROI, interpolation='nearest')
                # plt.show()
                faceROIResized = cv2.resize(faceROI, config['newSizeImage'], interpolation=cv2.INTER_AREA)
                # save the resulting frame as jpg
                iteratImagIndex = iteratImagIndex + 1
                iteratImagName = f'{base_string}_{iteratImagIndex:05}.jpg'
                destinationPathFile = os.path.join(destinationPath, iteratImagName)
                cv2.imwrite(destinationPathFile, faceROIResized)
                # augment image and save as jpg
                if config['augmentation']:
                    aug_img = augmentations.augment(faceROIResized)
                    destinationPathFileAug = os.path.join(destinationPath_augmen, iteratImagName)
                    cv2.imwrite(destinationPathFileAug, aug_img)
            else:
                warnings.warn('Warning Message: Face detection detected more than one face')
                noFaceList[iteratingFrames] = 1
        else:
            noFaceList[iteratingFrames] = 1


    return noFaceList
