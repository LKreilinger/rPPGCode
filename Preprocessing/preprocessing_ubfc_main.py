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

    if os.path.exists(config['datasetPath']) and os.path.isdir(config['datasetPath']):
        shutil.rmtree(config['datasetPath'])
    os.mkdir(config['datasetPath'])

    for idx, key in enumerate(config):
        # Video Preprocessing saving frames with detected face
        noFaceListAllVideos, delete_videos = video_preprocessing.video_pre_ubfc(config, key)

        # Save and load list with information about detected faces
        save_load.save_variable(config, noFaceListAllVideos, delete_videos, key)
        noFaceListAllVideos, delete_videos = save_load.load_variable(config, key)

        # Pulse preprocessing and fit to frames
        pulse_preprocessing_main.pulse_pre_ubfc(config, noFaceListAllVideos, delete_videos, key)

        # Make annotation.txt of the images and the pulse data
        path_dataset_split = os.path.join(config['datasetPath'], key)
        makeTxt.makeAnnotation(config['tempPathNofile'], path_dataset_split, config['nFramesVideo'])
        if idx == 2:
            break


