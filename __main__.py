"""
Change venv:    Run in Terminal -> .\venv\Scripts\activate
                if error -> first:
                Set-ExecutionPolicy Unrestricted -Scope Process

Install required packages
pip install -r .\pythonPackages.txt
pip uninstall
"""
import torch
# internal modules
from Preprocessing import preprocessing_ubfc_main, pre_config, config_dataset
from Preprocessing.WCD import PreprocessingWCDMain
from Preprocessing.PURE import preprocessing_pure_main
from cnn_process import train_rppg, train_pure, train_pure_rppg

if __name__ == '__main__':
    docker = True
    n_FRAMES_VIDEO = 128  # number of Frames used for training Model
    tempPathNofile, genPath, workingPath = pre_config.pre_config(docker)

    # Preprocessing datasets
    #config_pre_UBFC_Phys, config_pre_WCD, config_pre_UBFC_rPPG, config_pre_PURE = config_dataset.config_datasets(genPath, tempPathNofile, workingPath, n_FRAMES_VIDEO)
    #preprocessing_ubfc_main.pre_ubfc(config_pre_UBFC_Phys)
    # preprocessing_ubfc_main.pre_ubfc(config_pre_UBFC_rPPG)
    # PreprocessingWCDMain.preprocessing_wcd_dataset(config_pre_WCD)
    # preprocessing_pure_main.pre_pure(config_pre_PURE)

    # %% Complete cnn process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = 8  # batch size
    lr = 0.0001  # learning rate

    # 1 train rPPG                          -> test PURE and WCD (split subject and video)
    epochs = 7
    n = "1"  # 1 train rPPG
    augmentation = False
    train_rppg.train_rppg(genPath, augmentation, n_FRAMES_VIDEO, device, batch, lr, epochs, n)

    # 2 train rPPG with augment             -> test PURE and WCD (split subject and video)
    epochs = 37
    n = "2"  # 2 train rPPG with augment
    augmentation = True
    train_rppg.train_rppg(genPath, augmentation, n_FRAMES_VIDEO, device, batch, lr, epochs, n)

    # 3 train PURE                          -> test rPPG and WCD (split subject and video)
    epochs = 13
    n = "3"  # 3 train PURE
    train_pure.train_pure(genPath, n_FRAMES_VIDEO, device, batch, lr, epochs, n)

    # 4 train rPPG (augment) and PURE       -> test WCD (split subject and video)
    epochs = 9
    n = "4"  # 4 train rPPG (augment) and PURE
    train_pure_rppg.train_pure_rppg(genPath, n_FRAMES_VIDEO, device, batch, lr, epochs, n)






