# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:17:28 2022

@author: Chaputa
"""

import pandas
import scipy.signal
import numpy as np


def pulse_prepro(path: object, tempPath: object, SamplingRate: object, NewSamplingRate: object, noFaceList: object) -> object:
    
    df = pandas.read_csv(path)
    numberValuesOriginal = int(len(df.index))
    lengthBVP = int(numberValuesOriginal / SamplingRate) + 1
    NumberValues30 = int(lengthBVP * NewSamplingRate)
    NumberValues30ImageCount = len(noFaceList)
    if abs(NumberValues30ImageCount-NumberValues30) > 1:
        dif = NumberValues30ImageCount-NumberValues30
        print("Warning: Difference between image count and number of pulse data =", dif)
    tempNewDF = scipy.signal.resample(df, int(NumberValues30ImageCount))
    tempNewDF = np.round(tempNewDF, decimals=2)
    newDF = pandas.DataFrame(tempNewDF)
    
    for iterating in range(noFaceList.size):
        if noFaceList[iterating] == 1:
            newDF.drop(iterating, axis=0, inplace=True)
    
    newDF.to_csv(tempPath, index=False, header=False)
