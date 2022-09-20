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
from Preprocessing.PURE import preprocessing_pure_main

if __name__ == '__main__':
    docker = True
    n_FRAMES_VIDEO = 128  # number of Frames used fpr training Model
    tempPathNofile, genPath, workingPath = pre_config.pre_config(docker)
    config_pre_UBFC_Phys, config_pre_WCD, config_pre_UBFC_rPPG, config_pre_PURE = config_dataset.config_datasets(genPath, tempPathNofile, workingPath, n_FRAMES_VIDEO)

    # Preprocessing datasets
    #preprocessing_ubfc_main.pre_ubfc(config_pre_UBFC_Phys)
    #preprocessing_ubfc_main.pre_ubfc(config_pre_UBFC_rPPG)
    #PreprocessingWCDMain.preprocessing_wcd_dataset(config_pre_WCD)
    #preprocessing_pure_main.pre_pure(config_pre_PURE)

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


    # config_cnn_pure = dict(
    #     path_dataset=os.path.join(genPath, "output", "PURE_Dataset"),
    #     path_model=os.path.join(genPath, "output", "Model"),
    #     fps=30,
    #     nFramesVideo=n_FRAMES_VIDEO,
    #     device=device,
    #     epochs=10,
    #     batch_size=size,
    #     learning_rate=lr,
    #     dataset="UBFC_rPPG",
    #     architecture="PhysNet")
    #
    # model = cnn_process_main.cnn_process_main(config_cnn_pure)
    # # %% Test model with UBFC rPPG data
    # config_cnn_test_UBFC_rPPG = dict(
    #     path_dataset=os.path.join(genPath, "output", "UBFC_rPPG_Dataset", "test"),
    #     path_model=os.path.join(genPath, "output", "Model"),
    #     variblesPath=os.path.join(genPath, "output", "noFaceList"),
    #     nFramesVideo=n_FRAMES_VIDEO,
    #     fps=30,
    #     device=device,
    #     batch_size=size,
    #     subjects=2,
    #     dataset="WCD",
    #     architecture="PhysNet")
    #
    # test_wcd.test_model(config_cnn_test_UBFC_rPPG)
    #
    # # %% Test model with WCD data
    # config_cnn_test_wcd = dict(
    #     path_dataset=os.path.join(genPath, "output", "WCD_Dataset", "test"),
    #     path_model=os.path.join(genPath, "output", "Model"),
    #     variblesPath=os.path.join(genPath, "output", "noFaceList"),
    #     nFramesVideo=n_FRAMES_VIDEO,
    #     fps=30,
    #     device=device,
    #     batch_size=size,
    #     subjects=2,
    #     dataset="WCD",
    #     architecture="PhysNet")
    #
    # test_wcd.test_model(config_cnn_test_wcd)

    #%%
    config_cnn_ubfc_rppg = dict(
        path_dataset=os.path.join(genPath, "output", "UBFC_rPPG_Dataset"),
        path_model=os.path.join(genPath, "output", "Model"),
        fps=30,
        nFramesVideo=n_FRAMES_VIDEO,
        device=device,
        epochs=11,
        batch_size=size,
        learning_rate=lr,
        dataset="UBFC_rPPG",
        architecture="PhysNet")

    model = cnn_process_main.cnn_process_main(config_cnn_ubfc_rppg)
    # %% Test model with PURE data
    config_cnn_test_pure = dict(
        path_dataset=os.path.join(genPath, "output", "PURE_Dataset", "test"),
        path_model=os.path.join(genPath, "output", "Model"),
        variblesPath=os.path.join(genPath, "output", "noFaceList"),
        nFramesVideo=n_FRAMES_VIDEO,
        fps=30,
        device=device,
        batch_size=size,
        subjects=2,
        dataset="WCD",
        architecture="PhysNet")

    test_wcd.test_model(config_cnn_test_pure)

    # %% Test model with WCD data
    config_cnn_test_wcd = dict(
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

    test_wcd.test_model(config_cnn_test_wcd)
    # #%% combined rPPG and PURE
    # batch_sizes = [32, 16, 8, 4, 2]
    # learning_rates = [0.01, 0.001, 0.0001] #default 0.0001
    # for size in batch_sizes:
    #     for lr in learning_rates:
    # config_cnn_pure_and_UBFC_rPPG = dict(
    #     path_dataset=os.path.join(genPath, "output", "PURE_and_rPPG_Dataset"),
    #     path_model=os.path.join(genPath, "output", "Model"),
    #     fps=30,
    #     nFramesVideo=n_FRAMES_VIDEO,
    #     device=device,
    #     epochs=100,
    #     batch_size=size,
    #     learning_rate=lr,
    #     dataset="UBFC_rPPG_and_PURE",
    #     architecture="PhysNet")
    #
    # model = cnn_process_main.cnn_process_main(config_cnn_pure_and_UBFC_rPPG)
    # # %% Test model with WCD data
    # config_cnn_wcd = dict(
    #     path_dataset=os.path.join(genPath, "output", "WCD_Dataset", "test"),
    #     path_model=os.path.join(genPath, "output", "Model"),
    #     variblesPath=os.path.join(genPath, "output", "noFaceList"),
    #     nFramesVideo=n_FRAMES_VIDEO,
    #     fps=30,
    #     device=device,
    #     batch_size=size,
    #     subjects=2,
    #     dataset="WCD",
    #     architecture="PhysNet")
    #
    # test_wcd.test_model(config_cnn_wcd)
