import os


def config_datasets(genPath, tempPathNofile, workingPath, n_FRAMES_VIDEO):
    # Preprocessing UBFC_Phys Dataset
    randomSeed = 3
    config_pre_UBFC_Phys = dict(
        train_split=60,
        validation_split=15,
        test_split=25,
        randomSeed=randomSeed,
        samplingRatePulse=64,
        newSamplingRatePulse=30,
        newFpsVideo=30,
        newSizeImage=(128, 128),
        patternVideo="*.avi",
        patternPuls="*.csv",
        datasetPath=os.path.join(genPath, "output", "UBFC_Phys_Dataset"),
        genPathData=os.path.join(genPath, "data", "UBFC_Phys"),
        variblesPath=os.path.join(genPath, "output", "noFaceList"),
        tempPathNofile=tempPathNofile,
        workingPath=workingPath,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(90, 90),
        nFramesVideo=n_FRAMES_VIDEO)

    # Preprocessing WCD Dataset
    config_pre_WCD = dict(
        train=0,
        validation=0,
        test=100,
        randomSeed=randomSeed,
        samplingRatePulse=55,
        newSamplingRatePulse=30,
        newFpsVideo=30,
        newSizeImage=(128, 128),
        patternVideo="*.avi",
        patternPuls="*.csv",
        datasetPath=os.path.join(genPath, "output", "WCD_Dataset"),
        dataImages=os.path.join(genPath, "data", "WCD", "test", "data_Realsense"),
        dataPulse=os.path.join(genPath, "data", "WCD", "test", "data_Polar"),
        variblesPath=os.path.join(genPath, "output", "noFaceList"),
        tempPathNofile=tempPathNofile,
        workingPath=workingPath,
        scaleFactor=1.01,#1.03
        minNeighbors=1,#3
        minSize=(180, 180),#(85, 85)
        nFramesVideo=n_FRAMES_VIDEO)

    # Preprocessing UBFC_rPPG dataset
    config_pre_UBFC_rPPG = dict(
        train=90,
        validation=10,
        test=0,
        randomSeed=randomSeed,
        samplingRatePulse=30,
        newSamplingRatePulse=30,
        newFpsVideo=30,
        newSizeImage=(128, 128),
        patternVideo="*.avi",
        patternPuls="*.txt",
        datasetPath=os.path.join(genPath, "output", "UBFC_rPPG_Dataset"),
        genPathData=os.path.join(genPath, "data", "UBFC_rPPG"),
        variblesPath=os.path.join(genPath, "output", "noFaceList"),
        tempPathNofile=tempPathNofile,
        workingPath=workingPath,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(90, 90),
        nFramesVideo=n_FRAMES_VIDEO)
    return config_pre_UBFC_Phys, config_pre_WCD, config_pre_UBFC_rPPG