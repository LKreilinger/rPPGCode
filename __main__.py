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
from cnn_process import train_rppg, train_pure

if __name__ == '__main__':
    docker = True
    n_FRAMES_VIDEO = 128  # number of Frames used for training Model
    tempPathNofile, genPath, workingPath = pre_config.pre_config(docker)

    # Preprocessing datasets
    config_pre_UBFC_Phys, config_pre_WCD, config_pre_UBFC_rPPG, config_pre_PURE = config_dataset.config_datasets(genPath, tempPathNofile, workingPath, n_FRAMES_VIDEO)
    #preprocessing_ubfc_main.pre_ubfc(config_pre_UBFC_Phys)
    #preprocessing_ubfc_main.pre_ubfc(config_pre_UBFC_rPPG)
    #PreprocessingWCDMain.preprocessing_wcd_dataset(config_pre_WCD)
    #preprocessing_pure_main.pre_pure(config_pre_PURE)

    # %% Complete cnn process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = 8  # batch size
    lr = 0.0001  # learning rate

    # 1 train rPPG                          -> test PURE and WCD (split subject and video)
    epochs = 7
    n = "1"  # 1 train rPPG
    augmentation = False
    #train_rppg.train_rppg(genPath, augmentation, n_FRAMES_VIDEO, device, batch, lr, epochs, n)

    # 2 train rPPG with augment             -> test PURE and WCD (split subject and video)
    epochs = 37
    n = "2"  # 2 train rPPG with augment
    augmentation = True
    #train_rppg.train_rppg(genPath, augmentation, n_FRAMES_VIDEO, device, batch, lr, epochs, n)

    # 3 train PURE                          -> test rPPG and WCD (split subject and video)
    epochs = 15
    n = "3"  # 3 train PURE
    train_pure.train_pure(genPath, n_FRAMES_VIDEO, device, batch, lr, epochs, n)
    # 4 train rPPG (augment) and PURE       -> test WCD (split subject and video)







    #######################################################################
    #%% combined rPPG and PURE
    # batch_sizes = [8, 16]
    # learning_rates = [0.0001, 0.001, 0.01]
    # for size in batch_sizes:
    #     for lr in learning_rates:
    # #lr = 0.01 0.0001 0.001 0.01
    # #size = 16 32     32    32
    #         config_cnn_pure_and_UBFC_rPPG = dict(
    #             path_dataset=os.path.join(genPath, "output", "PURE_and_rPPG_Dataset"),
    #             path_model=os.path.join(genPath, "output", "Model"),
    #             fps=30,
    #             nFramesVideo=n_FRAMES_VIDEO,
    #             device=device,
    #             epochs=30,
    #             batch_size=size,
    #             learning_rate=lr,
    #             dataset="UBFC_rPPG_and_PURE",
    #             architecture="PhysNet")
    #
    #         model = cnn_process_main.cnn_process_main(config_cnn_pure_and_UBFC_rPPG)
    #         # Test model with WCD data
    #         config_cnn_wcd = dict(
    #             path_dataset=os.path.join(genPath, "output", "WCD_Dataset", "test"),
    #             path_model=os.path.join(genPath, "output", "Model"),
    #             variblesPath=os.path.join(genPath, "output", "noFaceList"),
    #             nFramesVideo=n_FRAMES_VIDEO,
    #             fps=30,
    #             device=device,
    #             batch_size=size,
    #             subjects=2,
    #             dataset="WCD",
    #             architecture="PhysNet")
    #
    #         test_wcd.test_model(config_cnn_wcd)
