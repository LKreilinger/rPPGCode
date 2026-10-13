"""
Shared test helpers. The tests need no dataset and no GPU: they write small synthetic recordings
and replace the Haar cascade face detector with one that always finds the same face.
"""
import datetime
import json
import os

import cv2
import numpy as np
import pytest
from dateutil.relativedelta import relativedelta

from Preprocessing import config_dataset

FRAME_SIZE = (160, 120)  # width, height
N_FRAMES = 270  # 9 seconds at 30 fps -> two clips of 128 frames


class FakeCascade:
    def __init__(self, *args, **kwargs):
        pass

    def detectMultiScale(self, gray, **kwargs):
        return np.array([[20, 10, 100, 100]])


@pytest.fixture
def fake_face_detector(monkeypatch):
    monkeypatch.setattr(cv2, "CascadeClassifier", FakeCascade)


@pytest.fixture
def base_dir(tmp_path, monkeypatch):
    # folder that contains data/ and output/; relative paths (wandb) stay inside of tmp_path
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    return str(tmp_path)


@pytest.fixture
def configs(base_dir):
    temp_path = os.path.join(base_dir, "temp")
    names = ["ubfc_phys", "wcd", "ubfc_rppg", "pure"]
    return dict(zip(names, config_dataset.config_datasets(base_dir, temp_path, base_dir, 128)))


def random_frame():
    return np.random.randint(0, 255, (FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8)


def pulse_signal(n_samples, sampling_rate, bpm=72):
    t = np.arange(n_samples) / sampling_rate
    return np.sin(2 * np.pi * bpm / 60 * t)


def make_ubfc_rppg(data_path, n_subjects):
    for s in range(1, n_subjects + 1):
        subject_path = os.path.join(data_path, "s" + str(s))
        os.makedirs(subject_path)
        writer = cv2.VideoWriter(os.path.join(subject_path, "vid_s" + str(s) + ".avi"),
                                 cv2.VideoWriter_fourcc(*"MJPG"), 30, FRAME_SIZE)
        for _ in range(N_FRAMES):
            writer.write(random_frame())
        writer.release()
        # ground truth of UBFC-rPPG: line 1 pulse signal, line 2 heart rate, line 3 time
        bvp = " ".join("%e" % value for value in pulse_signal(N_FRAMES, 30))
        with open(os.path.join(subject_path, "bvp_s" + str(s) + ".txt"), "w") as f:
            f.write(bvp + "\n" + bvp + "\n" + bvp + "\n")


def make_pure(data_path, n_subjects):
    for s in range(1, n_subjects + 1):
        subject = "%02d" % s
        image_path = os.path.join(data_path, subject, subject + "-01")
        os.makedirs(image_path)
        for i in range(N_FRAMES):
            cv2.imwrite(os.path.join(image_path, "Image%05d.png" % i), random_frame())
        waveform = pulse_signal(N_FRAMES * 2, 60) * 50 + 100
        package = [{"Timestamp": i, "Value": {"pulseRate": 72, "waveform": float(value)}}
                   for i, value in enumerate(waveform)]
        with open(os.path.join(data_path, subject, subject + "-01.json"), "w") as f:
            json.dump({"/FullPackage": package, "/Image": []}, f)


def make_wcd(data_path):
    image_path = os.path.join(data_path, "test", "data_Realsense", "s1", "T1")
    pulse_path = os.path.join(data_path, "test", "data_Polar", "s1", "T1")
    os.makedirs(image_path)
    os.makedirs(pulse_path)
    first_frame = datetime.datetime(2022, 7, 19, 10, 0, 0)
    for i in range(N_FRAMES):
        frame_time = first_frame + datetime.timedelta(seconds=i / 30)
        cv2.imwrite(os.path.join(image_path, str(int(frame_time.timestamp() * 1000)) + ".jpg"), random_frame())
    # the Polar sensor counts from the year 2000 and its clock is two hours ahead (see convertPolarData.py)
    polar_start = first_frame - datetime.timedelta(seconds=1) - relativedelta(years=30) + relativedelta(hours=2)
    n_samples = 55 * 12
    signal = pulse_signal(n_samples, 55) * 1000
    with open(os.path.join(pulse_path, "Polar_Sense_PPG.txt"), "w") as f:
        f.write("Phone timestamp;sensor timestamp [ns];channel 0;channel 1;channel 2;ambient\n")
        for i in range(n_samples):
            ns = int((polar_start + datetime.timedelta(seconds=i / 55)).timestamp() * 1e9)
            value = int(signal[i]) + 5000
            f.write("x;%d;%d;%d;%d;%d\n" % (ns, value, value, value, 100))


def read_annotations(split_path):
    with open(os.path.join(split_path, "annotations.txt")) as f:
        return [line.split() for line in f.read().splitlines()]
