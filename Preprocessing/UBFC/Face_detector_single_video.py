# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:20:35 2022

@author: Chaputa
"""

import cv2
import os
import numpy as np
from moviepy.editor import *


#load trained modul for face detection
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)



# Change fps to 30fps
# clip = VideoFileClip(r'C:\Users\Chaputa\Documents\Trier\Master\Masterarbeit\data\s2\vid_s2_T2.AVI') 
# fpsOriginal=clip.fps
# if fpsOriginal != 30:
#     clip.write_videofile(r'C:\Users\Chaputa\Documents\Trier\Master\Masterarbeit\code\temp\vid_s2_T2.AVI', fps=30, codec="libx264")
# o documentation availab
video_capture = cv2.VideoCapture(r'C:\Users\Chaputa\Documents\Trier\Master\Masterarbeit\code\temp\vid_s1_T2.AVI')
# fps frames per second from original video
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
length = int(video_capture.get(cv2. CAP_PROP_FRAME_COUNT))
noFaceList=np.zeros(length) # 0=face detectet; 1=no face detected
base_string='\img'
newsize = (128, 128)
#videoLength=2 # new "Face video" length in seconds
#maxFrames=int(length/128)
maxFrames=length
firstRectagle=0 # selecting first rectangele size for all, in one video
iteratImagIndex=0
for iteratingFrames in range(500, 900, 1):
        ret, frame = video_capture.read()
        #cv2.imshow('Video',frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.2,
                                             minNeighbors=10,
                                             minSize=(64, 64),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        
        # define size of new video
        if firstRectagle==0 and len(faces)!=0:
            firstRectagle=1
            (xF,yF,wF,hF)=faces.T
            xF=int(xF)
            yF=int(yF)
            wF=int(wF)
            hF=int(hF)
            frame_width = wF
            frame_height = hF
            
       # write video frame by frame    
        if len(faces)!=0:
            (x,y,w,h)=faces.T
            x=int(x)
            y=int(y)
            faceROI = frame[y:y+hF,x:x+wF]
            faceROIResized = cv2.resize(faceROI, newsize, interpolation = cv2.INTER_AREA)
            # save the resulting frame as png
            iteratImagIndex=iteratImagIndex+1
            iteratImagName=f'{base_string}_{iteratImagIndex:05}.png'
            cv2.imwrite(r'C:\Users\Chaputa\Documents\Trier\Master\Masterarbeit\code\temp'+iteratImagName, faceROIResized)
        else:
            noFaceList[iteratingFrames]=1
            cv2.imshow('Video',frame)
            
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video_capture.release()





