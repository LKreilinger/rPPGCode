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
from Preprocessing.UBFC import PreprocessingUBFCMain
from cnn_process import cnn_process_main
from Preprocessing.WCD import PreprocessingWCDMain

if __name__ == '__main__':
    # for docker change workdir
    docker = True
    if docker:
        print("Docker is working")
        workingPath = os.path.abspath(os.getcwd())
        genPath = workingPath
    else:
        workingPath = os.path.abspath(os.getcwd())
        genPath = os.path.dirname(workingPath)

    outputData = os.path.join(genPath, "output", )
    outputDataUBFCPath = os.path.join(outputData, "UBFCDataset")
    path_model = os.path.join(outputData, "Model")
    outputDataWCDPath = os.path.join(outputData, "WCDDataset")
    outputDataUBFCSplitPath = os.path.join(outputData, "UBFCDatasetSplit")
    outputDataWCDSplitPath = os.path.join(outputData, "WCDDatasetSplit")
    nFramesVideo = 128  # number of Frames used fpr training Model
    # %%
    #       Preprocessing UBFC_Phys Dataset

    #PreprocessingUBFCMain.preprocessing_ubfc_dataset(genPath, nFramesVideo, workingPath, docker)

    # %%
    # Preprocessing WCD Dataset
    # PreprocessingWCDMain.preprocessing_wcd_dataset(genPath, nFramesVideo)

    # %%
    # Preprocessing UBFC_rPPG dataset
    config_preprocessing_UBFC_rPPG = dict(
        path_dataset=outputDataUBFCPath,
        nFramesVideo=nFramesVideo)
    # %%
    # Complete deep neuronal network process including
    #   - split data
    #   - load data
    #   - define training environment
    #   - define model
    #   - train and validate model
    #   - evaluate model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_cnn = dict(
        path_dataset=outputDataUBFCPath,
        path_dataset_split=outputDataUBFCSplitPath,
        path_model=path_model,
        train_split=60,
        validation_split=15,
        test_split=25,
        nFramesVideo=nFramesVideo,
        device=device,
        epochs=20,
        batch_size=64,
        learning_rate=0.0001,
        dataset="UBFC",
        architecture="CNN")

    model = cnn_process_main.cnn_process_main(config_cnn)

    # WCD
    # splitData.split_data(outputDataWCDPath, nFramesVideo)
    # training_loader, validation_loader, test_loader = loadData.load_data(outputDataWCDSplitPath, nFramesVideo)
