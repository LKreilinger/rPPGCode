import wandb
# internal modules
from load import load_main
import splitData
from TrainValidate import trainMain

def model_pipeline(hyperparameters):
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # split data
        splitData.split_data(config)

        # make the model, data, and optimization problem
        model, train_loader, validation_loader, test_loader, loss_Inst, optimizer = load_main.load_main(config)
        print(model)

        # and use them to train the model
        trainMain.train_and_validate_model(model, train_loader, validation_loader, loss_Inst, optimizer, config)

        # and test its final performance
        test(model, test_loader)

    return model
