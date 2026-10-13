"""
Master Thesis
Laurens Kreilinger B. Sc.
Title: Deep learning-enabled remote monitoring of pulse rate for versatile patients

Install required packages
pip install -r .\requirements.txt

Show all commands and options
python __main__.py --help
"""
import argparse
import os
import torch
# internal modules
from Preprocessing import preprocessing_ubfc_main, pre_config, config_dataset, merge_datasets
from Preprocessing.WCD import PreprocessingWCDMain
from Preprocessing.PURE import preprocessing_pure_main
from cnn_process import train_rppg, train_pure, train_pure_rppg

n_FRAMES_VIDEO = 128  # number of Frames used for training Model
BATCH = 8  # batch size
LR = 0.0001  # learning rate

# Datasets that are preprocessed when no dataset is named
DEFAULT_DATASETS = ["ubfc_rppg", "pure", "wcd"]
DATASETS = DEFAULT_DATASETS + ["ubfc_phys"]

# Folders inside of the output folder each experiment needs: training (train and validation) and tests
EXPERIMENTS = {
    # 1 train UBFC-rPPG                     -> test PURE and WCD (split subject and video)
    "1": dict(epochs=7, train="UBFC_rPPG_Dataset_no_aug", test=["PURE_Dataset", "WCD_Dataset"]),
    # 2 train UBFC-rPPG with augment        -> test PURE and WCD (split subject and video)
    "2": dict(epochs=37, train="UBFC_rPPG_Dataset", test=["PURE_Dataset", "WCD_Dataset"]),
    # 3 train PURE                          -> test rPPG and WCD (split subject and video)
    "3": dict(epochs=13, train="PURE_Dataset", test=["UBFC_rPPG_Dataset_no_aug", "WCD_Dataset"]),
    # 4 train UBFC-rPPG (augment) and PURE  -> test WCD (split subject and video)
    "4": dict(epochs=9, train="PURE_and_rPPG_Dataset", test=["WCD_Dataset"]),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python __main__.py",
        description="rPPG pipeline: preprocess the datasets, train PhysNet and test it on the other datasets. "
                    "Without a command, the default datasets are preprocessed and all experiments are run.")
    parser.add_argument("--docker", action="store_true",
                        help="run inside of the docker container (data and output are in the working directory)")
    parser.add_argument("--base-dir",
                        help="folder that contains the data and the output folder "
                             "(default: parent folder of the working directory)")
    subparsers = parser.add_subparsers(dest="command")

    pre = subparsers.add_parser("preprocess", help="face detection, pulse resampling and annotations")
    pre.add_argument("datasets", nargs="*", choices=DATASETS, metavar="dataset",
                     help="one or more of: " + ", ".join(DATASETS) + " (default: " + ", ".join(DEFAULT_DATASETS) + ")")
    pre.add_argument("--split", nargs=3, type=int, metavar=("TRAIN", "VALIDATION", "TEST"),
                     help="share of the subjects per split in percent, e.g. --split 90 10 0")
    pre.add_argument("--augmentation", dest="augmentation", action="store_true", default=None,
                     help="also save an augmented copy of every video")
    pre.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    pre.add_argument("--output-name", help="name of the dataset folder inside of the output folder")

    merge = subparsers.add_parser("merge", help="copy several preprocessed datasets into one")
    merge.add_argument("inputs", nargs="+", help="names of dataset folders inside of the output folder")
    merge.add_argument("--output-name", required=True, help="name of the merged dataset folder")

    train = subparsers.add_parser("train", help="train and test the model")
    train.add_argument("experiments", nargs="*", choices=list(EXPERIMENTS), metavar="experiment",
                       help="one or more of: " + ", ".join(EXPERIMENTS) + " (default: all)")
    train.add_argument("--epochs", type=int, help="number of epochs (default: value used in the thesis)")
    train.add_argument("--batch-size", type=int, default=BATCH)
    train.add_argument("--lr", type=float, default=LR, help="learning rate")
    return parser.parse_args(argv)


def preprocess(args, genPath, tempPathNofile, workingPath):
    config_pre_UBFC_Phys, config_pre_WCD, config_pre_UBFC_rPPG, config_pre_PURE = config_dataset.config_datasets(
        genPath, tempPathNofile, workingPath, n_FRAMES_VIDEO)
    steps = {
        "ubfc_rppg": (preprocessing_ubfc_main.pre_ubfc, config_pre_UBFC_rPPG),
        "pure": (preprocessing_pure_main.pre_pure, config_pre_PURE),
        "wcd": (PreprocessingWCDMain.preprocessing_wcd_dataset, config_pre_WCD),
        "ubfc_phys": (preprocessing_ubfc_main.pre_ubfc, config_pre_UBFC_Phys),
    }
    datasets = getattr(args, "datasets", None) or DEFAULT_DATASETS
    for name in datasets:
        pre_function, config = steps[name]
        if getattr(args, "split", None):
            config["train"], config["validation"], config["test"] = args.split
        if getattr(args, "augmentation", None) is not None:
            config["augmentation"] = args.augmentation
        if getattr(args, "output_name", None):
            config["datasetPath"] = os.path.join(genPath, "output", args.output_name)
        print("Preprocessing", name, "->", config["datasetPath"])
        pre_function(config)


def check_experiments(genPath, experiments):
    # stop before a long training run if a preprocessed dataset is missing
    missing = []
    for n in experiments:
        train_path = os.path.join(genPath, "output", EXPERIMENTS[n]["train"])
        needed = [os.path.join(train_path, "train", "annotations.txt"),
                  os.path.join(train_path, "validation", "annotations.txt")]
        for test_dataset in EXPERIMENTS[n]["test"]:
            test_path = os.path.join(genPath, "output", test_dataset, "test")
            # the test folder of WCD may be divided in subfolders, each with its own annotations.txt
            needed.append(test_path if test_dataset == "WCD_Dataset" else os.path.join(test_path, "annotations.txt"))
        for path in needed:
            if not os.path.exists(path) and path not in missing:
                missing.append(path)
    if missing:
        raise SystemExit("Preprocessed data is missing:\n  " + "\n  ".join(missing) +
                         "\nCreate it with 'python __main__.py preprocess' (see README.md, Running the pipeline).")


def train(args, genPath):
    # %% Complete cnn process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = getattr(args, "batch_size", BATCH)
    lr = getattr(args, "lr", LR)
    experiments = getattr(args, "experiments", None) or list(EXPERIMENTS)
    check_experiments(genPath, experiments)
    for n in experiments:
        epochs = getattr(args, "epochs", None) or EXPERIMENTS[n]["epochs"]
        if n == "1":
            train_rppg.train_rppg(genPath, False, n_FRAMES_VIDEO, device, batch, lr, epochs, n)
        elif n == "2":
            train_rppg.train_rppg(genPath, True, n_FRAMES_VIDEO, device, batch, lr, epochs, n)
        elif n == "3":
            train_pure.train_pure(genPath, n_FRAMES_VIDEO, device, batch, lr, epochs, n)
        elif n == "4":
            train_pure_rppg.train_pure_rppg(genPath, n_FRAMES_VIDEO, device, batch, lr, epochs, n)


if __name__ == '__main__':
    args = parse_args()
    tempPathNofile, genPath, workingPath = pre_config.pre_config(args.docker, args.base_dir)

    if args.command in (None, "preprocess"):
        preprocess(args, genPath, tempPathNofile, workingPath)
    if args.command == "merge":
        merge_datasets.merge_datasets(os.path.join(genPath, "output"), args.inputs, args.output_name)
    if args.command in (None, "train"):
        train(args, genPath)
