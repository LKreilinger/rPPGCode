import torch
import numpy as np
import glob
import os
import heartpy as hp

# local Packages
from cnn_process.TrainValidate import get_pulse
from cnn_process.load.load_main import PhysNet


def test_model(config, test_loader):
    # load best (newest) model
    saved_model = PhysNet.PhysNet_padding_Encoder_Decoder_MAX()
    model_path = os.path.join(config.path_model, "*")
    files = glob.glob(model_path)
    best_model_path = max(files, key=os.path.getctime)
    saved_model.load_state_dict(torch.load(best_model_path, map_location=config.device))
    saved_model.eval()
    n = 0
    BVP_label_all = np.empty([])
    rPPG_all = np.empty([])
    with torch.no_grad():
        for data in test_loader:
            inputs, BVP_label = data
            # prepare data
            inputs = inputs.permute(0, 2, 1, 3, 4)  # [batch,channel,length,width,height] = x.shape
            # print(inputs.shape)
            BVP_label = torch.stack(BVP_label)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            rPPG, x_visual, x_visual3232, x_visual1616 = saved_model(inputs)
            if torch.cuda.is_available():
                rPPG = rPPG.cpu()

            rPPG = rPPG.permute(1, 0)  # [(nframs label), batch] = y.shape
            rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
            BVP_label = (BVP_label - torch.mean(BVP_label.float())) / torch.std(BVP_label.float())  # normalize

            rPPGNP = rPPG.detach().numpy()
            BVP_labelNP = BVP_label.detach().numpy()
            if n == 0:
                rPPG_all = np.add(rPPG_all, rPPGNP)
                BVP_label_all = np.add(BVP_label_all, BVP_labelNP)
                n = 1
            else:
                rPPG_all = np.concatenate((rPPG_all, rPPGNP), axis=1)
                BVP_label_all = np.concatenate((BVP_label_all, BVP_labelNP), axis=1)

        # get puls
        pulse_label = np.zeros((1, BVP_label_all.shape[1]))
        pulse_predic = np.zeros((1, BVP_label_all.shape[1]))
        for col in range(BVP_label_all.shape[1]):
            working_data_BVP_label_all, measures_BVP_label_all = hp.process(BVP_label_all[:, col], config.fps)
            pulse_label[0, col] = measures_BVP_label_all['bpm']
            working_data_rPPG_all, measures_rPPG_all = hp.process(rPPG_all[:, col], config.fps)
            pulse_predic[0, col] = measures_rPPG_all['bpm']



