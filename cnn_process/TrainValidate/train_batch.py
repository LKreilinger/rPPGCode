import torch


def train_batch(inputs, BVP_label, optimizer, model, loss_Inst):
    # Zero your gradients for every batch!
    optimizer.zero_grad()

    inputs = inputs.permute(0, 2, 1, 3, 4)  # [batch,channel,length,width,height] = x.shape
    # print(inputs.shape)
    BVP_label = torch.stack(BVP_label)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    rPPG, x_visual, x_visual3232, x_visual1616 = model(inputs)
    if torch.cuda.is_available():
        rPPG = rPPG.cpu()
    # Compute the loss and its gradients
    # Normalized the Predicted rPPG signal and GroundTruth BVP signal

    rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
    BVP_label = (BVP_label - torch.mean(BVP_label.float())) / torch.std(BVP_label.float())  # normalize

    # Calculate the loss
    loss = loss_Inst(rPPG, BVP_label)
    loss.backward()

    # Adjust learning weights
    optimizer.step()

    return loss
