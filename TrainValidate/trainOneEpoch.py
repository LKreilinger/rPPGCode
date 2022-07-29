import torch


def train_one_epoch(epoch_index, tb_writer, training_loader, optimizer, model, loss_Inst):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, BVP_label = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
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
        loss_ecg = loss_Inst(rPPG, BVP_label)
        loss_ecg.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss_ecg.item()
        if i % 10 == 9:
            last_loss = running_loss / 10  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
