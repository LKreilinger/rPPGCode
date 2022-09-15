
import torch
# internal modules
from cnn_process.load import PhysNet
from cnn_process.load import load_data
from cnn_process.load import lossFunction

def load_main(config):
    # Load the data
    train_loader, validation_loader = load_data.load_data(config)

    # Load the model
    model = PhysNet.PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(config.device)

    # Load the loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_Inst = lossFunction.pearson_correlatio()


    return model, train_loader, validation_loader, loss_Inst, optimizer