"""
Change venv:    Run in Terminal -> .\venv\Scripts\activate
                if error -> first:
                Set-ExecutionPolicy Unrestricted -Scope Process

Install required packages
pip install -r .\requirements.txt
pip uninstall
"""
import time
import os
# internal modules
from TestModel import Testmain
from TrainValidate import loadData, trainMain
from Preprocessing import splitData
from Preprocessing.UBFC import PreprocessingUBFCMain
from Preprocessing.WCD import PreprocessingWCDMain

if __name__ == '__main__':
    # for docker change workdir
    docker = False
    if docker:
        print("Docker is working")
        workingPath = os.path.abspath(os.getcwd())
        genPath = workingPath
        outputData = os.path.join(workingPath, "output",)
        outputDataUBFCPath = os.path.join(outputData, "UBFCDataset")
        outputDataWCDPath = os.path.join(outputData, "WCDDataset")
        outputDataUBFCSplitPath = os.path.join(outputData, "UBFCDatasetSplit")
        outputDataWCDSplitPath = os.path.join(outputData, "WCDDatasetSplit")
        model_path = os.path.join(outputData, "Model")
    else:
        workingPath = os.path.abspath(os.getcwd())
        genPath = os.path.dirname(workingPath)
        outputDataUBFCPath = os.path.join(genPath, "output", "UBFCDataset")
        outputDataWCDPath = os.path.join(genPath, "output", "WCDDataset")
        outputDataUBFCSplitPath = os.path.join(genPath, "output", "UBFCDatasetSplit")
        outputDataWCDSplitPath = os.path.join(genPath, "output", "WCDDatasetSplit")
        model_path = os.path.join(genPath, "output", "Model")
    nFramesVideo = 128  # number of Frames used fpr training Model
    #%%
    #       Preprocessing UBFC Dataset
    PreprocessingUBFCMain.preprocessing_ubfc_dataset(genPath, nFramesVideo, workingPath)


    #%%
    # Preprocessing WCD Dataset
    #PreprocessingWCDMain.preprocessing_wcd_dataset(genPath, nFramesVideo)


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
    test_loader = trainMain.train_model(outputDataUBFCPath, Plot_results, training_loader, validation_loader, test_loader)


    #%%
    #       Test PhysNet Model
    Plot_results = True

    Testmain.test_model(model_path, test_loader, Plot_results)
