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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # for docker change workdir
    docker = True
    if docker:
        print("Docker is working")
        workingPath = '\\Users\\Chaputa\\Documents\\Trier\\Master\\Masterarbeit\\rPPGCode'
        #while True:
        #    time.sleep(1)
        #    print('Still working')
    else:
        workingPath = os.path.abspath(os.getcwd())
    genPath = os.path.dirname(workingPath)
    outputDataUBFCPath = os.path.join(genPath, "output", "UBFCDataset")
    outputDataWCDPath = os.path.join(genPath, "output", "WCDDataset")
    nFramesVideo = 128  # number of Frames used fpr training Model
    #############################
    #       Preprocessing UBFC Dataset
    #PreprocessingUBFCMain.preprocessing_ubfc_dataset(genPath, nFramesVideo)
    #############################

    #############################
    # Preprocessing WCD Dataset
    #PreprocessingWCDMain.preprocessing_wcd_dataset(genPath, nFramesVideo)
    #############################

    #############################
    #       Split and load data
    # UBFC
    #splitData.split_data(outputDataUBFCPath, nFramesVideo)
    outputDataUBFCSplitPath = os.path.join(genPath, "output", "UBFCDatasetSplit")
    training_loader, validation_loader, test_loader = loadData.load_data(outputDataUBFCSplitPath, nFramesVideo)

    # WCD
    # splitData.split_data(outputDataWCDPath, nFramesVideo)
    # outputDataWCDSplitPath = os.path.join(genPath, "output", "WCDDatasetSplit")
    # training_loader, validation_loader, test_loader = loadData.load_data(outputDataWCDSplitPath, nFramesVideo)
    #############################

    #############################
    #       Train and validate with PhysNet Model
    Plot_results = True
    test_loader = trainMain.train_model(outputDataUBFCPath, Plot_results, training_loader, validation_loader, test_loader)
    #############################

    #############################
    #       Test PhysNet Model
    Plot_results = True
    model_path = os.path.join(genPath, "output", "Model")
    Testmain.test_model(model_path, test_loader, Plot_results)
    ############################
