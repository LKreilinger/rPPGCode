import os
# internal modules
from cnn_process import cnn_process_main
from cnn_process.TestModel import test_wcd


def train_rppg(genPath, augmentation, n_FRAMES_VIDEO, device, size, lr, epochs, n):
    if augmentation:
        dataset_name = "UBFC_rPPG_Dataset"
    else:
        dataset_name = "UBFC_rPPG_Dataset_no_aug"

    config_cnn_ubfc_rppg = dict(
        path_dataset=os.path.join(genPath, "output", dataset_name),
        path_model=os.path.join(genPath, "output", "Model", n),
        fps=30,
        nFramesVideo=n_FRAMES_VIDEO,
        device=device,
        epochs=epochs,
        batch_size=size,
        learning_rate=lr,
        dataset="UBFC_rPPG",
        architecture="PhysNet")

    model = cnn_process_main.cnn_process_main(config_cnn_ubfc_rppg)
    # Test model with PURE data
    config_cnn_test_pure = dict(
        path_dataset=os.path.join(genPath, "output", "PURE_Dataset", "test"),
        path_model=os.path.join(genPath, "output", "Model", n),
        variblesPath=os.path.join(genPath, "output", "noFaceList", n),
        nFramesVideo=n_FRAMES_VIDEO,
        fps=30,
        device=device,
        batch_size=size,
        subjects=2,
        dataset="WCD",
        architecture="PhysNet")

    test_wcd.test_model(config_cnn_test_pure)

    #  Test model with WCD data
    config_cnn_test_wcd = dict(
        path_dataset=os.path.join(genPath, "output", "WCD_Dataset", "test"),
        path_model=os.path.join(genPath, "output", "Model", n),
        variblesPath=os.path.join(genPath, "output", "noFaceList", n),
        nFramesVideo=n_FRAMES_VIDEO,
        fps=30,
        device=device,
        batch_size=size,
        subjects=2,
        dataset="WCD",
        architecture="PhysNet")
    list_split = os.listdir(config_cnn_test_wcd["path_dataset"])
    for folder in list_split:
        config_cnn_test_wcd["path_dataset"] = os.path.join(config_cnn_test_wcd["path_dataset"], folder)
        config_cnn_test_wcd["dataset"] = "WCD_" + folder
        test_wcd.test_model(config_cnn_test_wcd)
        config_cnn_test_wcd["path_dataset"] = os.path.join(genPath, "output", "WCD_Dataset", "test")