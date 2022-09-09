import os
import shutil
# internal modules
from Preprocessing.WCD import image_preprocessing, pulse_pre_wcd_main
from Preprocessing import makeTxt, save_load, split_data


def preprocessing_wcd_dataset(config):
    # if os.path.exists(config['datasetPath']) and os.path.isdir(config['datasetPath']):
    #     shutil.rmtree(config['datasetPath'])
    # os.mkdir(config['datasetPath'])

    for idx, split in enumerate(config):
        if config[split] != 0:
            # Video Preprocessing saving frames with detected face
            # noFaceListAllVideos, delete_videos = image_preprocessing.image_pre_WCD(config, split)
            # Save and load list with information about detected faces
            # save_load.save_variable(config, noFaceListAllVideos, delete_videos, split)
            noFaceListAllVideos, delete_videos = save_load.load_variable(config, split)

            # Pulse preprocessing and fit to frames
            pulse_pre_wcd_main.pulse_wcd(config, noFaceListAllVideos, delete_videos, split)

            # Make annotation.txt of the images and the pulse data
            path_dataset_split = os.path.join(config['datasetPath'], split)
            makeTxt.makeAnnotation(config['tempPathNofile'], path_dataset_split, config['nFramesVideo'])
        if idx == 2:
            break
