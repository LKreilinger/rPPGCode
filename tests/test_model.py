import os

import numpy as np
import pytest
import torch

from Preprocessing import preprocessing_ubfc_main
from cnn_process import cnn_process_main
from cnn_process.TestModel import performance_metrics, test_wcd
from cnn_process.load import PhysNet, lossFunction
from conftest import make_ubfc_rppg
from old_py_files import get_pulse  # the version used by performance_metrics


def test_pulse_rate_from_fft():
    fps = 30
    t = np.arange(300) / fps
    signal = np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz
    assert get_pulse.get_rfft_pulse(signal, fps) == pytest.approx(90)


def test_metrics_are_zero_for_a_perfect_prediction():
    fps = 30
    t = np.arange(300) / fps
    # one column per clip
    signals = np.stack([np.sin(2 * np.pi * f * t) for f in (1.0, 1.5, 2.0)], axis=1)
    config = type("Config", (), {"fps": fps})
    MAE, RMSE, R = performance_metrics.eval_model_fft(signals, signals, config)
    assert MAE == 0 and RMSE == 0
    assert R == pytest.approx(1)


def test_neg_pearson_loss():
    loss = lossFunction.Neg_Pearson()
    x = torch.randn(4, 128)
    assert float(loss(x, x)) == pytest.approx(0, abs=1e-4)
    assert float(loss(x, -x)) == pytest.approx(2, abs=1e-4)


def test_physnet_returns_one_value_per_frame():
    model = PhysNet.PhysNet_padding_Encoder_Decoder_MAX(frames=128).eval()
    with torch.no_grad():
        rPPG = model(torch.rand(1, 3, 128, 128, 128))[0]  # [batch, channel, frames, height, width]
    assert rPPG.shape == (1, 128)


@pytest.mark.slow
def test_train_and_test_one_epoch(configs, base_dir, fake_face_detector, monkeypatch):
    # preprocessing -> training -> saved model -> test, on the CPU and without a Weights & Biases account
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    config = configs["ubfc_rppg"]
    make_ubfc_rppg(config["genPathData"], n_subjects=4)
    config.update(train=50, validation=25, test=25, augmentation=False)
    preprocessing_ubfc_main.pre_ubfc(config)

    path_model = os.path.join(base_dir, "output", "Model", "test")
    cnn_process_main.cnn_process_main(dict(
        path_dataset=config["datasetPath"], path_model=path_model, fps=30, nFramesVideo=128,
        device=torch.device("cpu"), epochs=1, batch_size=2, learning_rate=0.0001,
        dataset="UBFC_rPPG", architecture="PhysNet"))
    assert len(os.listdir(path_model)) == 1

    path_results = os.path.join(base_dir, "output", "noFaceList", "test")
    test_wcd.test_model_wcd(dict(
        path_dataset=os.path.join(config["datasetPath"], "test"), path_model=path_model,
        variblesPath=path_results, nFramesVideo=128, fps=30, device=torch.device("cpu"),
        batch_size=2, subjects=2, dataset="UBFC_rPPG", architecture="PhysNet"))
    assert sorted(os.listdir(path_results)) == ["BVP_labelUBFC_rPPG.pkl", "rPPG_predictUBFC_rPPG.pkl"]
