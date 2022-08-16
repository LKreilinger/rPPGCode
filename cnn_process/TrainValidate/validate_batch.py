
import torch

def val_batch(inputs, BVP_label, model, loss_Inst):
    # prepare data
    inputs = inputs.permute(0, 2, 1, 3, 4)  # [batch,channel,length,width,height] = x.shape
    # print(inputs.shape)
    BVP_label = torch.stack(BVP_label)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    rPPG, x_visual, x_visual3232, x_visual1616 = model(inputs)
    if torch.cuda.is_available():
        rPPG = rPPG.cpu()
    # Calculate the loss
    rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
    BVP_label = (BVP_label - torch.mean(BVP_label.float())) / torch.std(BVP_label.float())  # normalize
    validation_loss_ecg = loss_Inst(rPPG, BVP_label)
    validation_loss_ecg.backward()

    return validation_loss_ecg, rPPG