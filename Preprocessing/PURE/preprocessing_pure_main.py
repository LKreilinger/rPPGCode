import os
import shutil
# internal modules
from Preprocessing import save_load, makeTxt, split_data
from Preprocessing.PURE import image_preprocessing_main, pulse_pre_pure_main


def pre_pure(config):
    # Change folder name and file name of the original data. Run only once for the UBFC_rPPG dataset
    #change_names.change_filename_foldername(config['genPathData'])

    # split data
    split_data.split_data(config)

    if os.path.exists(config['datasetPath']) and os.path.isdir(config['datasetPath']):
        shutil.rmtree(config['datasetPath'])
    os.mkdir(config['datasetPath'])

    for idx, split in enumerate(config):
        if config[split] != 0:
            # Video Preprocessing saving frames with detected face
            noFaceListAllVideos, delete_videos = image_preprocessing_main.image_pre_pure(config, split)
            # Save and load list with information about detected faces
            save_load.save_variable(config, noFaceListAllVideos, delete_videos, split)
            noFaceListAllVideos, delete_videos = save_load.load_variable(config, split)

            # Pulse preprocessing and fit to frames
            pulse_pre_pure_main.pulse_pure(config, noFaceListAllVideos, delete_videos, split)
            # Make annotation.txt of the images and the pulse data
            path_dataset_split = os.path.join(config['datasetPath'], split)
            makeTxt.makeAnnotation(config['tempPathNofile'], path_dataset_split, config['nFramesVideo'])
        if idx == 2:
            break
