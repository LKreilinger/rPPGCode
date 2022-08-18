# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:15:03 2022

@author: Chaputa
"""

import os
import pandas

#%% Generate annotations.txt in dataset
#   folder/videoname.png 1 17 0
#   Path/Name; start Frame; end frame; label (pulsdata)
# Number of Frames per video -> 128

def makeAnnotation(tempPath, Datasetpath, nFramesVideo):
    destinationPath=os.path.join(Datasetpath, "annotations.txt")
    f = open(destinationPath, "w")   # 'r' for reading and 'w' for writing
    for path, subdirs, files in os.walk(Datasetpath):
        if files != [] and 'annotations.txt' not in files:
            nOfElements = len(files)
            numberVideos=int(nOfElements/nFramesVideo)
            Name=os.path.basename(os.path.normpath(path))
            tempPathAName=os.path.join(tempPath,  Name)
            temPvpFile=tempPathAName.replace("vid", "bvp")
            temPvpFile=temPvpFile.replace("avi", "csv")
            dfLabels = pandas.read_csv(temPvpFile, header=None)
            NPLabels=dfLabels.to_numpy()
                
            for videoSplit in range(numberVideos):
                StartFrame=str(videoSplit*nFramesVideo+1)
                EndFrame=str((videoSplit+1)*nFramesVideo)
                LabelsSplit=NPLabels[videoSplit*nFramesVideo:(videoSplit+1)*nFramesVideo].T
                #LabelsSplit=LabelsSplit.astype(str)
                f.write(Name+ " " + StartFrame + " " + EndFrame + " ")    # Write inside file
                for listitem in range(len(LabelsSplit[0])):
                    single_label=str(int(LabelsSplit[0, listitem]))
                    f.write(single_label + " ")
                f.write("\n") 
    f.close()
