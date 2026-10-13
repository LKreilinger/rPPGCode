import os

import numpy as np
import pandas

from Preprocessing import preprocessing_ubfc_main, pulse_adjust_fps_no_face, split_data, merge_datasets
from Preprocessing.PURE import preprocessing_pure_main
from Preprocessing.WCD import PreprocessingWCDMain
from conftest import make_ubfc_rppg, make_pure, make_wcd, read_annotations


def check_split(split_path, n_videos, clips_per_video=2):
    lines = read_annotations(split_path)
    assert len(lines) == n_videos * clips_per_video
    for line in lines:
        video, start, end = line[0], int(line[1]), int(line[2])
        assert len(line) == 3 + 128  # name, first frame, last frame and one pulse value per frame
        assert end - start + 1 == 128
        assert os.path.isfile(os.path.join(split_path, video, "img_%05d.jpg" % end))


def test_configs_start_with_the_three_splits(configs):
    # split_data and the preprocessing loops read the first three entries as the splits
    for config in configs.values():
        assert list(config)[:3] == ["train", "validation", "test"]
        assert "augmentation" in config


def test_split_data(configs):
    config = configs["ubfc_rppg"]
    config.update(train=60, validation=20, test=20)
    for s in range(10):
        os.makedirs(os.path.join(config["genPathData"], "s" + str(s)))
    split_data.split_data(config)
    counts = {split: len(os.listdir(os.path.join(config["genPathData"], split)))
              for split in ("train", "validation", "test")}
    assert counts == {"train": 6, "validation": 2, "test": 2}
    # a second run undoes the first split
    config.update(train=90, validation=10, test=0)
    split_data.split_data(config)
    assert sorted(os.listdir(config["genPathData"])) == ["train", "validation"]
    assert len(os.listdir(os.path.join(config["genPathData"], "train"))) == 9


def test_pulse_is_resampled_and_frames_without_face_are_removed(tmp_path):
    path = str(tmp_path / "bvp.csv")
    pandas.DataFrame(np.arange(641)).to_csv(path, index=False, header=False)
    no_face_list = np.zeros(300)
    no_face_list[:10] = 1
    pulse_adjust_fps_no_face.pulse_prepro(path, path, 64, 30, no_face_list)
    assert len(pandas.read_csv(path, header=None)) == 290


def test_preprocess_ubfc_rppg_with_augmentation(configs, fake_face_detector):
    config = configs["ubfc_rppg"]
    make_ubfc_rppg(config["genPathData"], n_subjects=2)
    config.update(train=50, validation=50, test=0)
    preprocessing_ubfc_main.pre_ubfc(config)
    for split in ("train", "validation"):
        split_path = os.path.join(config["datasetPath"], split)
        # every video once as recorded and once augmented
        check_split(split_path, n_videos=2)
        assert len([name for name in os.listdir(split_path) if name.startswith("vid_a_s")]) == 1
    assert not os.path.exists(os.path.join(config["datasetPath"], "test"))


def test_preprocess_pure(configs, fake_face_detector):
    config = configs["pure"]
    make_pure(config["genPathData"], n_subjects=4)
    config.update(train=50, validation=0, test=50)
    preprocessing_pure_main.pre_pure(config)
    check_split(os.path.join(config["datasetPath"], "train"), n_videos=2)
    check_split(os.path.join(config["datasetPath"], "test"), n_videos=2)


def test_preprocess_wcd(configs, fake_face_detector):
    config = configs["wcd"]
    make_wcd(os.path.dirname(os.path.dirname(config["dataImages"])))
    PreprocessingWCDMain.preprocessing_wcd_dataset(config)
    check_split(os.path.join(config["datasetPath"], "test"), n_videos=1)


def test_lists_of_other_datasets_are_kept(configs, fake_face_detector):
    make_ubfc_rppg(configs["ubfc_rppg"]["genPathData"], n_subjects=2)
    make_pure(configs["pure"]["genPathData"], n_subjects=2)
    configs["ubfc_rppg"].update(train=50, validation=50, test=0, augmentation=False)
    configs["pure"].update(train=50, validation=50, test=0)
    preprocessing_ubfc_main.pre_ubfc(configs["ubfc_rppg"])
    preprocessing_pure_main.pre_pure(configs["pure"])
    for name in ("ubfc_rppg", "pure"):
        assert sorted(os.listdir(configs[name]["variblesPath"])) == [
            "delete_videos_train.pkl", "delete_videos_validation.pkl",
            "noFaceListAllVideos_train.pkl", "noFaceListAllVideos_validation.pkl"]


def test_merge_datasets(configs, base_dir, fake_face_detector):
    make_ubfc_rppg(configs["ubfc_rppg"]["genPathData"], n_subjects=2)
    make_pure(configs["pure"]["genPathData"], n_subjects=2)
    configs["ubfc_rppg"].update(train=50, validation=50, test=0, augmentation=False)
    configs["pure"].update(train=50, validation=50, test=0)
    preprocessing_ubfc_main.pre_ubfc(configs["ubfc_rppg"])
    preprocessing_pure_main.pre_pure(configs["pure"])
    output_path = os.path.join(base_dir, "output")
    merge_datasets.merge_datasets(output_path, ["PURE_Dataset", "UBFC_rPPG_Dataset"], "PURE_and_rPPG_Dataset")
    for split in ("train", "validation"):
        check_split(os.path.join(output_path, "PURE_and_rPPG_Dataset", split), n_videos=2)
