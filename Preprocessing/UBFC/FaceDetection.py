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

def viola_jonas_face_detector(currentPath: str, destinationPath: str, tempPath: str, NewSamplingRate: int,
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
    #faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

    # Change fps to NewSamplingRate
    clip = VideoFileClip(currentPath)
    fpsOriginal = clip.fps
    # %%
    #clip = clip.subclip(0, 5)  # !!!!! only first 12 seconds!!!!
    # %%
    if fpsOriginal != NewSamplingRate:
        clip.write_videofile(tempPath, fps=NewSamplingRate, codec="libx264")
    else:
        # !!!!! only first 12 seconds!!!!
        #clip = VideoFileClip(currentPath)
        #clip = clip.subclip(0, 5)
        #clip.write_videofile(tempPath, fps=NewSamplingRate, codec="libx264")

        # copy file to tempPath
        shutil.copyfile(currentPath, tempPath)

    video_capture = cv2.VideoCapture(tempPath)
    # fps frames per second from original video
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    noFaceList = np.zeros(length)  # 0=face detectet; 1=no face detected
    base_string = "img"
    # videoLength=2 # new "Face video" length in seconds
    # maxFrames=int(length/128)
    maxFrames = length
    firstRectagle = 0  # selecting first rectangele size for all, in one video
    iteratImagIndex = 0

    print("tempPath", tempPath)
    print("currentPath", currentPath)
    print("destinationPath", destinationPath)

    for iteratingFrames in range(maxFrames):
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=6,
                                             minSize=(90, 90),
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

                faceROIResized = cv2.resize(faceROI, newsizeImage, interpolation=cv2.INTER_AREA)
                # save the resulting frame as png
                iteratImagIndex = iteratImagIndex + 1
                iteratImagName = f'{base_string}_{iteratImagIndex:05}.jpg'
                destinationPathFile = os.path.join(destinationPath, iteratImagName)
                cv2.imwrite(destinationPathFile, faceROIResized)
            else:
                warnings.warn('Warning Message: Face detection detected more than one face')
                noFaceList[iteratingFrames] = 1
        else:

            noFaceList[iteratingFrames] = 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    clip.close()

    return noFaceList
