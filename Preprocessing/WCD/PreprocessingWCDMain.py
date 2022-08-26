# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:15:28 2022

@author: Laurens Kreilinger
"""
import os
import shutil
import pickle
import numpy as np

# internal modules
from Preprocessing.WCD import convertPolarData, image_preprocessing, pulse_pre_wcd_main

from Preprocessing import makeTxt, pulse_adjust_fps_no_face, save_load


def preprocessing_wcd_dataset(config) -> None:
    """
    Main Funktion for preprocessing the UBFC_Phys Dataset
    :rtype: None
    :param gen_path:
    """



    if os.path.exists(config['datasetPath']) and os.path.isdir(config['datasetPath']):
        shutil.rmtree(config['datasetPath'])
    os.mkdir(config['datasetPath'])


    # Video Preprocessing saving frames with detected face
    noFaceListAllVideos, delete_videos = image_preprocessing.image_pre_WCD(config)
    data_split = "test_WCD"
    # Save and load list with information about detected faces
    save_load.save_variable(config, noFaceListAllVideos, delete_videos, data_split)
    save_load.load_variable(config, data_split)

    # Pulse preprocessing and fit to frames
    pulse_pre_wcd_main.pulse_wcd(config, noFaceListAllVideos, delete_videos)

    # Make annotation.txt of the images and the pulse data
    makeTxt.makeAnnotation(config['tempPathNofile'], config['datasetPath'], config['nFramesVideo'])
