# Deep learning-enabled remote monitoring of pulse rate for versatile patients

[![Tests](https://github.com/LKreilinger/rPPGCode/actions/workflows/tests.yml/badge.svg)](https://github.com/LKreilinger/rPPGCode/actions/workflows/tests.yml)

Source code of the master thesis "Deep learning-enabled remote monitoring of pulse rate for versatile patients".

The code extracts a pulse signal from face videos with deep learning (remote photoplethysmography, rPPG). It preprocesses several rPPG datasets, trains a [PhysNet](https://arxiv.org/abs/1905.02419) model on them, and tests each trained model on the other datasets (cross-dataset validation).

Want to improve it? See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and a list of open tasks.

## Contents

- [Abstract](#abstract)
- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Preparing the data](#preparing-the-data)
- [Running the pipeline](#running-the-pipeline)
- [Running with Docker](#running-with-docker)
- [Results and outputs](#results-and-outputs)
- [Tests](#tests)
- [Recording your own data](#recording-your-own-data)
- [Known limitations](#known-limitations)
- [License](#license)
- [Cite as](#cite-as)

## Abstract

A method of data collection has been devised that uses a mobile and remote photoplethysmography (PPG) setup to obtain image data from people monitored with an RGB camera and pulse sensor. Through deep learning, the pulse rate (PR) is determined in a contactless manner using a self-generated dataset and two publicly available datasets. This study shows that the PhysNet model performs well on its training dataset, which is the UBFC-rPPG dataset. However, validating deep learning models with cross-dataset in this study indicates a weakness in the generalizability of the models. In summary, the method proposed for remote-photoplethysmography (rPPG) is yet to be suitable for the reliable determination of PR in versatile patients.

## How it works

Everything is started from [`__main__.py`](__main__.py), which has two stages:

1. **Preprocessing** (`Preprocessing/`)
   - Splits the subjects of a dataset into `train`, `validation` and `test`.
   - Detects the face in every frame (Viola-Jones / Haar cascade), crops it and resizes it to 128 x 128 pixels at 30 fps.
   - Resamples the reference pulse signal to 30 Hz and removes the samples that belong to frames without a detected face.
   - Optionally writes an augmented copy of each video (flip, rotation, darker with more contrast).
   - Writes an `annotations.txt` that cuts every video into clips of 128 frames with the matching pulse values.
2. **Training and testing** (`cnn_process/`)
   - Trains PhysNet with the negative Pearson loss and the Adam optimizer (batch size 8, learning rate 0.0001).
   - After each epoch, calculates the pulse rate error on the validation split (MAE, RMSE, Pearson R).
   - Tests the trained model on the other datasets.
   - Logs everything to [Weights & Biases](https://wandb.ai).

The four experiments of the thesis:

| No. | Training data | Dataset folder | Epochs | Tested on |
| --- | ------------- | -------------- | ------ | --------- |
| 1 | UBFC-rPPG | `UBFC_rPPG_Dataset_no_aug` | 7 | PURE, WCD |
| 2 | UBFC-rPPG with augmentation | `UBFC_rPPG_Dataset` | 37 | PURE, WCD |
| 3 | PURE | `PURE_Dataset` | 13 | UBFC-rPPG, WCD |
| 4 | UBFC-rPPG with augmentation and PURE | `PURE_and_rPPG_Dataset` | 9 | WCD |

WCD is the dataset recorded for this thesis with an Intel RealSense camera and a Polar pulse sensor.

## Repository structure

```text
__main__.py            Command line entry point
Preprocessing/         Face detection, pulse resampling, splitting, annotations
  config_dataset.py    Paths and settings for every dataset
  pre_config.py        Base paths (local run or Docker)
  merge_datasets.py    Joins preprocessed datasets
  PURE/, UBFC_rPPG/, UBFC_Phys/, WCD/   Dataset-specific steps
  Augmentation/        Image augmentation
cnn_process/           Model, data loading, training, validation, testing
  load/PhysNet.py      Network architecture
  train_rppg.py, train_pure.py, train_pure_rppg.py   One file per experiment
tests/                 Automated tests on synthetic data
data_collection/       Scripts used to record the WCD dataset
old_py_files/          Older scripts (get_pulse.py is still used for the test metrics)
Dockerfile, docker-compose.yml, build_image.sh        Docker setup
requirements.txt       Python packages for preprocessing and training
requirements-data-collection.txt   Extra packages for data_collection/
requirements-dev.txt   Extra packages for development (pytest)
LICENSE, THIRD_PARTY_NOTICES.md      License of this code and of third-party code
```

## Requirements

- Python 3.10, 3.11 or 3.12 (tested with 3.12; `pykale` does not support 3.13 yet)
- An NVIDIA GPU with CUDA is strongly recommended. The code falls back to the CPU, but training 3D convolutions on 128-frame clips is very slow there.
- A free [Weights & Biases](https://wandb.ai) account (or offline mode, see below)
- The datasets, which are not part of this repository:
  - [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)
  - [PURE](https://www.tu-ilmenau.de/neurob/data-sets-code/pulse-rate-detection-dataset-pure)
  - [UBFC-Phys](https://sites.google.com/view/ybenezeth/ubfc-phys) (optional, not used by the four experiments)
  - WCD (self-recorded, see [Recording your own data](#recording-your-own-data))

## Installation

```bash
git clone https://github.com/LKreilinger/rPPGCode.git
cd rPPGCode

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

On Windows and macOS this installs the CPU build of PyTorch. For an NVIDIA GPU on Windows, install the CUDA build **before** `requirements.txt`, otherwise pip keeps the CPU build (see [pytorch.org](https://pytorch.org/get-started/previous-versions/) for other CUDA versions):

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

On Linux, `requirements.txt` already installs the CUDA build.

The versions in `requirements.txt` are pinned because the packages limit each other: `pykale` 0.2.0 needs `torch` 2.6.0 or older and `pandas` 2.2.2 or older, the Haar cascade face detector needs OpenCV 4, and `heartpy` needs `setuptools` below 81.

Then log in to Weights & Biases once:

```bash
wandb login
```

To train without an account and without uploading anything, set the environment variable `WANDB_MODE=offline` instead.

## Preparing the data

The code needs one base folder that contains a `data` folder with the raw datasets. It writes everything it produces to an `output` folder next to it:

```text
<base folder>/
├── data/
│   ├── UBFC_rPPG/
│   ├── PURE/
│   ├── WCD/
│   └── UBFC_Phys/     optional
└── output/            created by the code
```

Pass the base folder with `--base-dir`. Without it, the parent folder of the working directory is used, so `data` and `output` are expected next to the repository.

> **Warning:** Preprocessing moves the subject folders inside `data/<dataset>/` into `train/`, `validation/` and `test/` subfolders, and it deletes and rebuilds the dataset folder it writes to in `output/`. Keep a backup of the original downloads.

### Expected layout per dataset

**UBFC-rPPG**: one folder per subject, files named after the subject:

```text
data/UBFC_rPPG/
├── s1/
│   ├── vid_s1.avi
│   └── bvp_s1.txt
├── s3/
│   ├── vid_s3.avi
│   └── bvp_s3.txt
...
```

The original download uses other names. `Preprocessing/UBFC_rPPG/change_names.py` renames the subject folders to `s<N>`, files starting with `v` to `vid_s<N>.avi` and files starting with `b` to `bvp_s<N>.txt`. Run it once on the dataset (the call is commented out in `Preprocessing/preprocessing_ubfc_main.py`).

**PURE**: one folder per subject, containing one `.json` file and one image folder per recording:

```text
data/PURE/
├── 01/
│   ├── 01-01.json
│   ├── 01-01/         images of recording 01
│   ├── 01-02.json
│   ├── 01-02/
│   ...
├── 02/
...
```

**WCD**: camera frames and pulse data in two parallel trees. Frames are named with their timestamp in milliseconds; the pulse file must contain `PPG` in its name:

```text
data/WCD/test/
├── data_Realsense/
│   └── s1/
│       └── T1/        1658218108598.jpg, ...
└── data_Polar/
    └── s1/
        └── T1/        *PPG*.txt
```

**UBFC-Phys**: `data/UBFC_Phys/` with the `vid_*.avi` videos and the matching `bvp_*.csv` files.

### Settings

The defaults for every dataset are in [`Preprocessing/config_dataset.py`](Preprocessing/config_dataset.py). The split, the augmentation and the name of the output folder can also be set on the command line.

| Setting | Meaning |
| ------- | ------- |
| `train`, `validation`, `test` | Share of subjects per split in percent |
| `augmentation` | Also save an augmented copy of each video |
| `samplingRatePulse` | Sampling rate of the reference pulse signal in Hz |
| `newFpsVideo`, `newSamplingRatePulse` | Target rate for video and pulse (30) |
| `newSizeImage` | Size of the cropped face image (128 x 128) |
| `scaleFactor`, `minNeighbors`, `minSize` | Parameters of the face detector |
| `datasetPath`, `genPathData` | Where the preprocessed and the raw data are stored |

## Running the pipeline

Run the commands from the repository root. `python __main__.py --help` and `python __main__.py <command> --help` list all options.

| Command | What it does |
| ------- | ------------ |
| `python __main__.py preprocess [dataset ...]` | Preprocesses `ubfc_rppg`, `pure`, `wcd` and/or `ubfc_phys` (default: the first three) |
| `python __main__.py merge <name> <name> --output-name <name>` | Copies preprocessed datasets into one |
| `python __main__.py train [experiment ...]` | Trains and tests experiment `1`, `2`, `3` and/or `4` (default: all) |
| `python __main__.py` | Preprocesses the default datasets, then runs all experiments |

Put `--base-dir <folder>` directly after `__main__.py` if your data is not next to the repository.

### Example: experiment 2

Train on UBFC-rPPG with augmentation, test on PURE and WCD:

```bash
# training data: UBFC-rPPG, 90 % train, 10 % validation, with augmentation (the defaults)
python __main__.py --base-dir D:/rppg preprocess ubfc_rppg

# test data: all PURE subjects, and WCD
python __main__.py --base-dir D:/rppg preprocess pure --split 0 0 100
python __main__.py --base-dir D:/rppg preprocess wcd

python __main__.py --base-dir D:/rppg train 2
```

### Data for the other experiments

Before it starts, `train` checks that the dataset folders of the chosen experiments exist and names the missing ones. These commands create them:

```bash
# experiments 1 and 3: UBFC-rPPG without augmentation
python __main__.py preprocess ubfc_rppg --no-augmentation --output-name UBFC_rPPG_Dataset_no_aug

# experiment 4: PURE and augmented UBFC-rPPG, both preprocessed with a train and validation split
python __main__.py merge PURE_Dataset UBFC_rPPG_Dataset --output-name PURE_and_rPPG_Dataset
```

An experiment needs the `train` and `validation` split of its training dataset and the `test` split of the datasets it is tested on. One dataset is used for training in one experiment and for testing in another, so choose the split with `--split` to match the experiment you run. The splits used for the test sets in the thesis are not recorded in the code.

Training options: `--epochs`, `--batch-size` and `--lr`. Without them, the values of the thesis are used.

## Running with Docker

The Docker setup is written for a Linux server with an NVIDIA GPU and the NVIDIA Container Toolkit. The image is based on `python:3.12-slim`; the PyTorch package brings its own CUDA libraries.

1. Build the image:

   ```bash
   bash build_image.sh
   ```

2. In `docker-compose.yml`, set `volumes` to the host paths of your `data` and `output` folders. They are mounted as `/data` and `/output`.
3. Make `UID` and `GID` available to Docker Compose, for example in a `.env` file next to `docker-compose.yml`, and start the container:

   ```bash
   docker compose up -d
   docker compose exec kreilinger bash
   ```

4. Inside the container, run the pipeline with `--docker`:

   ```bash
   cd /
   wandb login          # or set WANDB_API_KEY
   python3 __main__.py --docker preprocess ubfc_rppg
   python3 __main__.py --docker train 2
   ```

## Results and outputs

| Path | Content |
| ---- | ------- |
| `output/<Dataset>_Dataset/<split>/<video>/img_00001.jpg ...` | Cropped face frames |
| `output/<Dataset>_Dataset/<split>/annotations.txt` | Clip list with pulse labels |
| `output/noFaceList/<dataset>/` | Frames without a detected face and removed videos (`.pkl`) |
| `output/Model/<n>/model_<timestamp>` | Trained weights of experiment `<n>` |
| `output/noFaceList/<n>/BVP_label<dataset>.pkl`, `rPPG_predict<dataset>.pkl` | Reference and predicted pulse signals of the test runs |

Training loss, validation loss and the pulse rate metrics (MAE, RMSE, Pearson R) are printed to the console and logged to the Weights & Biases project `pytorch-demo`. Testing always loads the newest model file in `output/Model/<n>/`.

If `output/WCD_Dataset/test/` contains an `annotations.txt`, the model is tested on all of it. To report results per subject or recording, move the videos into subfolders, each with its own `annotations.txt`; the model is then tested on every subfolder.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                  # everything, about 2 minutes on a CPU
pytest -m "not slow"    # skip the test that trains the model
```

The tests need no dataset, no GPU and no Weights & Biases account. They write small synthetic recordings, replace the face detector with one that always finds a face, and run preprocessing, one training epoch and a test run. They run automatically on GitHub for every push and pull request.

## Recording your own data

`data_collection/` contains the scripts used to record the WCD dataset on Windows with an Intel RealSense camera. They need a few more packages:

```bash
pip install -r requirements-data-collection.txt
```

- `RealsenseGetFrames.py` shows the live stream and saves 3 minutes of frames at 30 fps, named by timestamp.
- `PACED_test.py` plays the audio files of a paced arithmetic test (level normal or hard) while recording.

Before use, change the hard-coded paths `destination_path` (in `RealsenseGetFrames.py`) and `path_wav` (in `PACED_test.py`). The audio files are not part of this repository. Start both scripts at the same time from the `data_collection` folder:

```bat
start python PACED_test.py "&" start python RealsenseGetFrames.py
```

Press `1` (normal) or `2` (hard) to start recording and the test, and `c` to stop.

## Known limitations

- The splits that were used for the cross-dataset tests of the thesis are not recorded in the code, so the published numbers cannot be reproduced exactly from this repository alone.
- There are two versions of the pulse rate function. All metrics use `old_py_files/get_pulse.py` (search from 0.67 Hz); `cnn_process/TrainValidate/get_pulse.py` (from 0.9 Hz) is not used.
- The scripts in `data_collection/` and several helper scripts contain absolute paths from the author's machine. `data_collection/__main.py` does not work; start the two recording scripts as described above.
- The standalone scripts `Preprocessing/WCD/ConvertFramesToVideo.py` and `Preprocessing/UBFC_Phys/Face_detector_single_video.py` are not used by the pipeline and still use the MoviePy 1 API.
- The Docker image has not been rebuilt since the base image was changed to Python 3.12.

## License

The code written for this project is released under the [MIT License](LICENSE).

Two files are not covered by it: the PhysNet model and its loss function (`cnn_process/load/PhysNet.py`, `cnn_process/load/lossFunction.py`) come from Zitong Yu et al. and are for research only; commercial use is not allowed. The pipeline depends on both files, so in practice **this project can only be used for non-commercial research**. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details and for the other third-party code.

The datasets have their own terms of use.

## Cite as

Laurens Kreilinger. "Deep learning-enabled remote monitoring of pulse rate for versatile patients". Master Thesis. Trier: Trier University of Applied Sciences and Rosenheim: Rosenheim University of Applied Sciences. 2022
