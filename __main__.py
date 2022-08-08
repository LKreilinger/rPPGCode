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
from cnn_process.TestModel import Testmain
from cnn_process.TrainValidate import trainMain
from Preprocessing.UBFC import PreprocessingUBFCMain
from cnn_process import cnn_process_main, splitData

if __name__ == '__main__':
    # for docker change workdir
    docker = True
    if docker:
        print("Docker is working")
        workingPath = os.path.abspath(os.getcwd())
        genPath = workingPath
        outputData = os.path.join(workingPath, "output",)
        outputDataUBFCPath = os.path.join(outputData, "UBFCDataset")
        path_model = os.path.join(outputData, "Model")
        outputDataWCDPath = os.path.join(outputData, "WCDDataset")
        outputDataUBFCSplitPath = os.path.join(outputData, "UBFCDatasetSplit")
        outputDataWCDSplitPath = os.path.join(outputData, "WCDDatasetSplit")
        model_path = os.path.join(outputData, "Model")
    else:
        workingPath = os.path.abspath(os.getcwd())
        genPath = os.path.dirname(workingPath)
        outputDataUBFCPath = os.path.join(genPath, "output", "UBFCDataset")
        path_model = os.path.join(genPath, "output", "Model")
        outputDataWCDPath = os.path.join(genPath, "output", "WCDDataset")
        outputDataUBFCSplitPath = os.path.join(genPath, "output", "UBFCDatasetSplit")
        outputDataWCDSplitPath = os.path.join(genPath, "output", "WCDDatasetSplit")
        model_path = os.path.join(genPath, "output", "Model")

    nFramesVideo = 128  # number of Frames used fpr training Model
    #%%
    #       Preprocessing UBFC Dataset
    config_preprocessing = dict(
        path_dataset=outputDataUBFCPath,
        train_split=60,
        validation_split=15,
        test_split=25,
        nFramesVideo=nFramesVideo,
        epochs=5,
        batch_size=32,
        learning_rate=0.001,
        dataset="UBFC",
        architecture="CNN")
    PreprocessingUBFCMain.preprocessing_ubfc_dataset(genPath, nFramesVideo, workingPath, docker)


    #%%
    # Preprocessing WCD Dataset
    #PreprocessingWCDMain.preprocessing_wcd_dataset(genPath, nFramesVideo)

    #%%
    # Complete deep neuronal network process including
    #   - split data
    #   - load data
    #   - define training environment
    #   - define model
    #   - train and validate model
    #   - evaluate model

    # Device configuration
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
        epochs=5,
        batch_size=32,
        learning_rate=0.001,
        dataset="UBFC",
        architecture="CNN")

    cnn_process_main.cnn_process_main(config_cnn)

    #%%
    #       Split and load data
    # UBFC
    splitData.split_data(outputDataUBFCPath, nFramesVideo)
    training_loader, validation_loader, test_loader = loadData.load_data(outputDataUBFCSplitPath, nFramesVideo)
    
    # WCD
    # splitData.split_data(outputDataWCDPath, nFramesVideo)
    # training_loader, validation_loader, test_loader = loadData.load_data(outputDataWCDSplitPath, nFramesVideo)


    #%%
    #       Train and validate with PhysNet Model
    Plot_results = True
    test_loader = trainMain.train_and_validate_model(outputDataUBFCPath, Plot_results, training_loader,
                                                     validation_loader, test_loader)


    #%%
    #       Test PhysNet Model
    Plot_results = True

    Testmain.test_model(model_path, test_loader, Plot_results)
