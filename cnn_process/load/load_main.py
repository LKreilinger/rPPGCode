
import torch
# internal modules
import PhysNet
import load_data
import lossFunction

def load_main(config):
    # Load the data
    config.pa
    train_loader, validation_loader, test_loader = load_data.load_data(config)

    # Load the model
    model = PhysNet.PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(config.device)

    # Load the loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_Inst = lossFunction.Neg_Pearson()


    return model, train_loader, test_loader, validation_loader, loss_Inst, optimizer