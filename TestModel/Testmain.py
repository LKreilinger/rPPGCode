import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# local Packages
from TrainValidate import get_pulse
from TrainValidate import PhysNet

def test_model(model_path, test_loader, Plot_results):
    # load best (newest) model
    saved_model = PhysNet.PhysNet_padding_Encoder_Decoder_MAX()
    model_path = os.path.join(model_path, "*")
    files = glob.glob(model_path)
    best_model_path = max(files, key=os.path.getctime)
    device = torch.device('cpu')
    saved_model.load_state_dict(torch.load(best_model_path, map_location=device))
    saved_model.eval()
    n = 0
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

            if Plot_results:
                fps = 30
                rPPGNP = rPPG.detach().numpy()
                rPPGNP = np.transpose(rPPGNP)
                BVP_labelNP = BVP_label.detach().numpy()
                pulse_BVP_labelNP = get_pulse.get_rfft_pulse(BVP_labelNP, fps)  # get pulse from signal
                pulse_PPGNP = get_pulse.get_rfft_pulse(rPPGNP, fps)  # get pulse from signal
                max_time = rPPGNP.size / fps
                time_steps = np.linspace(0, max_time, rPPGNP.size)
                plt.figure(figsize=(15, 15))
                plt.title('EPOCH {}:'.format(n + 1))
                plt.plot(time_steps, rPPGNP, label='rPPG')
                plt.plot(time_steps, BVP_labelNP, label='BVP_label')
                plt.xlabel("Time [s]")
                plt.ylabel("Amplitude")
                plt.legend()
                plt.show()
                print('Label Puls {} Test result {}'.format(pulse_BVP_labelNP, pulse_PPGNP))
                n = n + 1
