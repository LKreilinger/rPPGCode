# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:18:27 2022
"""

import cv2
import os
from moviepy.editor import *
import numpy as np
import shutil
import warnings

def viola_jonas_face_detector_img(currentPath: str, destinationPath: str,
                                  newsizeImage: tuple, config) -> np.ndarray:
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
    :param NewSamplingRate:
    :param newsizeImage:
    """
    # load trained modul for face detection
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    file_count = 0
    for path in os.listdir(currentPath):
        # check if current path is a file
        if os.path.isfile(os.path.join(currentPath, path)):
            file_count += 1

    noFaceList = np.zeros(file_count)  # 0=face detectet; 1=no face detected
    base_string = "img"
    firstRectagle = 0  # selecting first rectangele size. Use this size for all the other
    iteratImagIndex = 0
    iterating = 0
    directory = sorted(os.listdir(currentPath))
    for filename in directory:
        # # only first n frames
        # if iterating == 129:
        #     break
        #read image and convert to ndarray (h,w,d) d->rgb
        file_path = os.path.join(currentPath, filename)
        frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
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

                noFaceList[iterating] = 1

        # write frame by frame
        if len(faces) != 0:
            (x, y, w, h) = faces.T
            if x.__len__() == 1:
                x = int(x)
                y = int(y)
                faceROI = frame[y:y + hF, x:x + wF]
                # # only to show single faceROI
                # from matplotlib import pyplot as plt
                # faceROI = frame[y[0]:y[0] + hF, x[0]:x[0] + wF]
                # #faceROI = frame[y:y + hF, x:x + wF]
                # plt.imshow(faceROI, interpolation='nearest')
                # plt.show()

                faceROIResized = cv2.resize(faceROI, newsizeImage, interpolation=cv2.INTER_AREA)
                # save the resulting frame as png
                iteratImagIndex = iteratImagIndex + 1
                iteratImagName = f'{base_string}_{iteratImagIndex:05}.jpg'
                destinationPathFile = os.path.join(destinationPath, iteratImagName)
                cv2.imwrite(destinationPathFile, faceROIResized)
            else:
                #print('Warning Message: Face detection detected more than one face')
                noFaceList[iterating] = 1

        else:
            noFaceList[iterating] = 1
        iterating = iterating + 1

    return noFaceList
