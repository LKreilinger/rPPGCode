import torch


def val_batch(inputs, BVP_label, model, loss_Inst):
    # prepare data
    # print(inputs.shape)
    BVP_label = torch.stack(BVP_label)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    inputs = inputs.permute(0, 2, 1, 3,
                            4).contiguous()  # [batch,channel,length,width,height] = x.shape .contiguous() memory
    # torch.backends.cudnn.enabled = False
    rPPG, x_visual, x_visual3232, x_visual1616 = model(inputs)
    if torch.cuda.is_available():
        rPPG = rPPG.cpu()
    # Calculate the loss
    rPPG = rPPG.permute(1, 0)  # [(nframs label), batch] = y.shape
    rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
    BVP_label = (BVP_label - torch.mean(BVP_label.float())) / torch.std(BVP_label.float())  # normalize
    validation_loss_ecg = loss_Inst(rPPG, BVP_label)
    del inputs
    del BVP_label
    torch.cuda.empty_cache()
    return validation_loss_ecg, rPPG
