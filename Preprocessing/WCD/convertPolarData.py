# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:17:28 2022

@author: Chaputa
"""

import pandas
import scipy.signal
import numpy as np
import datetime
from dateutil.relativedelta import *
import os

# internal modules


def convert_polar_data(path: object, tempPath: object, camera_data_path: str) -> object:
    #%% open file and save each column
    f = open(path, 'r')
    lines = f.readlines()
    lines.pop(0)

    polar_timestamp_ns = []
    Channel0 = []
    Channel1 = []
    Channel2 = []
    Ambient = []
    for line in lines:
        polar_timestamp_ns.append(line.split(';')[1])
        Channel0.append(line.split(';')[2])
        Channel1.append(line.split(';')[3])
        Channel2.append(line.split(';')[4])
        Ambient.append(line.split(';')[5])
    f.close()

    #%% synchronize polar data and Realsense data
    directory = sorted(os.listdir(camera_data_path))
    first_frame_ms = int(os.path.splitext(directory[0])[0])
    last_frame_ms = int(os.path.splitext(directory[-1])[0])
    first_frame_time = datetime.datetime.fromtimestamp(first_frame_ms / 1000)
    last_frame_time = datetime.datetime.fromtimestamp(last_frame_ms / 1000)
    int(polar_timestamp_ns[-1])
    date_polar_epoch_hour = datetime.datetime.fromtimestamp(int(polar_timestamp_ns[-1]) / 1000000000)
    counter = 0
    only_once = 0
    for ns in polar_timestamp_ns:
        ns = int(ns)
        date_polar_epoch_hour = datetime.datetime.fromtimestamp(ns / 1000000000)
        # todo change epoch possible mistake in wintertime
        date_polar_hour = date_polar_epoch_hour + relativedelta(years=+30)
        date_polar = date_polar_hour + relativedelta(hours=-2)

        # get a ploar timestamp which is the closest to first frame timestamp
        if date_polar > first_frame_time and only_once == 0:
            only_once = 1
            if abs(date_polar - first_frame_time) > abs(date_polar_temp - first_frame_time):
                delete_until = counter + 1  # date_polar_temp closer
            else:
                delete_until = counter  # date_polar closer
        # get a ploar timestamp which is the closest to last frame timestamp
        if date_polar > last_frame_time:
            if abs(date_polar - last_frame_time) > abs(date_polar_temp - last_frame_time):
                delete_form = counter + 1
                break
            else:
                delete_form = counter
                break
        date_polar_temp = date_polar
        counter = counter + 1
    # only to check if synchronisation is correct
    # del polar_timestamp_ns[:delete_until]
    # del polar_timestamp_ns[(delete_form + 1 - delete_until):]
    del Channel0[:delete_until]
    del Channel0[(delete_form + 1 - delete_until):]
    del Channel1[:delete_until]
    del Channel1[(delete_form + 1 - delete_until):]
    del Channel2[:delete_until]
    del Channel2[(delete_form + 1 - delete_until):]
    del Ambient[:delete_until]
    del Ambient[(delete_form + 1 - delete_until):]


    #%% Preprocess data to one vector
    # converting list to array
    Channel0 = np.array(Channel0)
    Channel1 = np.array(Channel1)
    Channel2 = np.array(Channel2)
    Ambient = np.array(Ambient)
    Channel0 = Channel0.astype(np.int32)
    Channel1 = Channel1.astype(np.int32)
    Channel2 = Channel2.astype(np.int32)
    Ambient = Ambient.astype(np.int32)

    PPG = (Channel0 + Channel1 + Channel2 - (3 * Ambient))/3
    PPG = np.round(PPG, decimals=0)

    #%% save PPG as *.csv
    newDF = pandas.DataFrame(PPG)
    newDF.to_csv(tempPath, index=False, header=False)


