import os
import shutil
# internal modules
from Preprocessing import video_preprocessing, save_load, pulse_preprocessing_main, makeTxt, split_data
from Preprocessing.UBFC_rPPG import change_names


def pre_ubfc(config):
    # Change folder name and file name of the original data. Run only once for the UBFC_rPPG dataset
    #change_names.change_filename_foldername(config['genPathData'])

    # split data
    split_data.split_data(config)
    data_splits = ["train", "validation", "test"]

    if os.path.exists(config['datasetPath']) and os.path.isdir(config['datasetPath']):
        shutil.rmtree(config['datasetPath'])
    os.mkdir(config['datasetPath'])

    for data_split in data_splits:
        # Video Preprocessing saving frames with detected face
        noFaceListAllVideos, delete_videos = video_preprocessing.video_pre_ubfc(config, data_split)

        # Save and load list with information about detected faces
        save_load.save_variable(config, noFaceListAllVideos, delete_videos, data_split)
        save_load.load_variable(config, data_split)

        # Pulse preprocessing and fit to frames
        pulse_preprocessing_main.pulse_pre_ubfc(config, noFaceListAllVideos, delete_videos, data_split)

        # Make annotation.txt of the images and the pulse data
        path_dataset_split = os.path.join(config['datasetPath'], data_split)
        makeTxt.makeAnnotation(config['tempPathNofile'], path_dataset_split, config['nFramesVideo'])


