"""
Change venv:    Run in Terminal -> .\venv\Scripts\activate
                if error -> first:
                Set-ExecutionPolicy Unrestricted -Scope Process

Install required packages
pip install -r .\pythonPackages.txt
pip uninstall
"""
import os
import torch
# internal modules
from cnn_process import cnn_process_main
from Preprocessing import preprocessing_ubfc_main

if __name__ == '__main__':
    # for docker change workdir
    docker = True
    if docker:
        print("Docker is working")
        workingPath = os.path.abspath(os.getcwd())
        genPath = workingPath
        tempPathNofile = os.path.join(workingPath, "output", "temp")
    else:
        print("Docker is NOT working")
        workingPath = os.path.abspath(os.getcwd())
        genPath = os.path.dirname(workingPath)
        tempPathNofile = os.path.join(workingPath, "temp")


    outputDataWCDPath = os.path.join(genPath, "output", "WCDDataset")
    outputDataWCDSplitPath = os.path.join(genPath, "output", "WCDDatasetSplit")
    n_FRAMES_VIDEO = 128  # number of Frames used fpr training Model
    # %%
    #       Preprocessing UBFC_Phys Dataset
    # Preprocessing UBFC_rPPG dataset
    config_pre_UBFC_Phys = dict(
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
    #preprocessing_ubfc_main.pre_ubfc(config_pre_UBFC_Phys)

    # %%
    # Preprocessing WCD Dataset
    # PreprocessingWCDMain.preprocessing_wcd_dataset(genPath, n_FRAMES_VIDEO)

    # %%
    # Preprocessing UBFC_rPPG dataset
    config_pre_UBFC_rPPG = dict(
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

    preprocessing_ubfc_main.pre_ubfc(config_pre_UBFC_rPPG)



    # %% UBFC_Phys
    # Complete cnn process
    #   - split data
    #   - load data
    #   - define training environment
    #   - define model
    #   - train and validate model
    #   - evaluate model
    batch_sizes = [4, 8, 16, 32]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for size in batch_sizes:
        config_cnn_ubfc_phys = dict(
            path_dataset=os.path.join(genPath, "output", "UBFC_Phys_Dataset"),
            path_dataset_split=os.path.join(genPath, "output", "UBFC_Phys_Dataset_Split"),
            path_model=os.path.join(genPath, "output", "Model"),
            train_split=60,
            validation_split=15,
            test_split=25,
            nFramesVideo=n_FRAMES_VIDEO,
            device=device,
            epochs=40,
            batch_size=size,
            learning_rate=0.0001,
            dataset="UBFC_Phys",
            architecture="PhysNet")

        config_cnn_ubfc_rppg = dict(
            path_dataset=os.path.join(genPath, "output", "UBFC_rPPG_Dataset"),
            path_dataset_split=os.path.join(genPath, "output", "UBFC_rPPG_Dataset_Split"),
            path_model=os.path.join(genPath, "output", "Model"),
            train_split=60,
            validation_split=15,
            test_split=25,
            nFramesVideo=n_FRAMES_VIDEO,
            device=device,
            epochs=40,
            batch_size=size,
            learning_rate=0.0001,
            dataset="UBFC_rPPG",
            architecture="PhysNet")


        model = cnn_process_main.cnn_process_main(config_cnn_ubfc_rppg)
        # model = cnn_process_main.cnn_process_main(config_cnn_ubfc_phys)
