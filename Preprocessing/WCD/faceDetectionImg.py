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
                                  newsizeImage: tuple) -> np.ndarray:
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
    #cascPathface = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_alt2.xml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
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

        # read image and convert to ndarray (h,w,d) d->rgb
        file_path = os.path.join(currentPath, filename)
        frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # todo What to do if two or more faces are detected
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.03,
                                             minNeighbors=3,
                                             minSize=(85, 85),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

        # define size of new image
        if firstRectagle == 0 and len(faces) != 0:
            firstRectagle = 1
            (xF, yF, wF, hF) = faces.T
            wF = int(wF)
            hF = int(hF)

        # write frame by frame
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

                faceROIResized = cv2.resize(faceROI, newsizeImage, interpolation=cv2.INTER_AREA)
                # save the resulting frame as png
                iteratImagIndex = iteratImagIndex + 1
                iteratImagName = f'{base_string}_{iteratImagIndex:05}.jpg'
                cv2.imwrite(destinationPath + iteratImagName, faceROIResized)
                destinationPathFile = os.path.join(destinationPath, iteratImagName)
                cv2.imwrite(destinationPathFile, faceROIResized)
            else:
                warnings.warn('Warning Message: Face detection detected more than one face')
                noFaceList[iterating] = 1

        else:

            noFaceList[iterating] = 1
        iterating = iterating + 1
        if iterating == 1000:
            break

    return noFaceList
