import wandb
# internal modules
from cnn_process.load import load_main
from cnn_process import splitData
from cnn_process.TrainValidate import trainMain
from cnn_process.TestModel import Testmain

def model_pipeline(hyperparameters):
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, validation_loader, loss_Inst, optimizer = load_main.load_main(config)
        print(model)

        # and use them to train the model
        trainMain.train_and_validate_model(model, train_loader, validation_loader, loss_Inst, optimizer, config)

        # and test its final performance
        #Testmain.test_model(config, test_loader)

    return model
