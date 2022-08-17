
# internal modules
from Preprocessing import video_preprocessing, save_load, pulse_preprocessing_main, makeTxt
from Preprocessing.UBFC_rPPG import change_names


def pre_ubfc_rppg(config):
    # Change folder name and file name of the original data. Run only once
    #change_names.change_filename_foldername(config['genPathData'])
    noFaceListAllVideos, delete_videos = video_preprocessing.video_pre_ubfc(config)

    save_load.save_variable(config, noFaceListAllVideos, delete_videos)
    save_load.load_variable(config)

    pulse_preprocessing_main.pulse_pre_ubfc(config, noFaceListAllVideos, delete_videos)

    makeTxt.makeAnnotation(config['tempPathNofile'], config['datasetPath'], config['nFramesVideo'])


