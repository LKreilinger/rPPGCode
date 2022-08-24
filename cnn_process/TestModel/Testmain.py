import torch
import numpy as np
import glob
import os
# local Packages
from cnn_process.TestModel import performance_metrics, append_matrix
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

            rPPG_all, BVP_label_all, first_run = append_matrix.append_truth_prediction_label(
                BVP_label, rPPG, first_run, rPPG_all, BVP_label_all)

        # Calculate performace of model with test data
        MAE, MSE = performance_metrics.eval_model(BVP_label_all, rPPG_all, config)
        print(f"Test MAE: {MAE:.3f}" + f" Test MSE: {MSE:.3f}")




