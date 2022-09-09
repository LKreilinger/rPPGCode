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
from Preprocessing import preprocessing_ubfc_main, pre_config, config_dataset
from Preprocessing.WCD import PreprocessingWCDMain
from cnn_process.TestModel import test_wcd

if __name__ == '__main__':
    # for docker change workdir
    docker = True
    n_FRAMES_VIDEO = 128  # number of Frames used fpr training Model
    tempPathNofile, genPath, workingPath = pre_config.pre_config(docker)
    config_pre_UBFC_Phys, config_pre_WCD, config_pre_UBFC_rPPG = config_dataset.config_datasets(genPath, tempPathNofile, workingPath, n_FRAMES_VIDEO)

    # Preprocessing
    #preprocessing_ubfc_main.pre_ubfc(config_pre_UBFC_Phys)
    #preprocessing_ubfc_main.pre_ubfc(config_pre_UBFC_rPPG)
    #PreprocessingWCDMain.preprocessing_wcd_dataset(config_pre_WCD)

    # %% Complete cnn process
    #   - load data
    #   - define training environment
    #   - define model
    #   - train and validate model
    #   - evaluate model
    # batch_sizes = [4, 8, 16, 32]
    # learning_rates = [0.01, 0.001, 0.0001] #default 0.0001
    size = 8
    learning_rates = [0.0001, 0.001]
    lr = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # for size in batch_sizes:
    # for lr in learning_rates:

    config_cnn_ubfc_phys = dict(
        path_dataset=os.path.join(genPath, "output", "UBFC_Phys_Dataset"),
        path_model=os.path.join(genPath, "output", "Model"),
        test_split=25,
        fps=30,
        nFramesVideo=n_FRAMES_VIDEO,
        device=device,
        epochs=40,
        batch_size=size,
        learning_rate=lr,
        dataset="UBFC_Phys",
        architecture="PhysNet")

    config_cnn_ubfc_rppg = dict(
        path_dataset=os.path.join(genPath, "output", "UBFC_rPPG_Dataset"),
        path_model=os.path.join(genPath, "output", "Model"),
        test_split=25,
        fps=30,
        nFramesVideo=n_FRAMES_VIDEO,
        device=device,
        epochs=15,
        batch_size=size,
        learning_rate=lr,
        dataset="UBFC_rPPG",
        architecture="PhysNet")

    model = cnn_process_main.cnn_process_main(config_cnn_ubfc_rppg)
    # model = cnn_process_main.cnn_process_main(config_cnn_ubfc_phys)
    # %% Test model with WCD data
    config_cnn_wcd = dict(
        path_dataset=os.path.join(genPath, "output", "WCD_Dataset", "test"),
        path_model=os.path.join(genPath, "output", "Model"),
        variblesPath=os.path.join(genPath, "output", "noFaceList"),
        nFramesVideo=n_FRAMES_VIDEO,
        fps=30,
        device=device,
        batch_size=size,
        subjects=2,
        dataset="WCD",
        architecture="PhysNet")

    test_wcd.test_model(config_cnn_wcd)
