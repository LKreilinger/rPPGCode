import wandb
import os
import pickle
# internal modules
from cnn_process.TestModel import load_test_data, Testmain
from cnn_process.TestModel import performance_metrics


def test_model(parameters):
    # load data and model
    wandb.login()
    with wandb.init(project="pytorch-demo", config=parameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        test_loader = load_test_data.load_test_data(config)
        BVP_label_all, rPPG_all = Testmain.test_model(config, test_loader)

        # for T in range(config.subject * 3 * 3):
        #     try:
        #         MAE, RMSE, STD = performance_metrics.eval_model(BVP_label_all[:,T], rPPG_all[:,T], config)
        #         print(f"Validation MAE: {MAE:.3f}" + f" Validation RMSE: {RMSE:.3f}")
        #     except Exception:
        #         print("Could not determine pulse for given signal.")

        name_BVP_label_all = "BVP_label_all.pkl"
        list_path = config.variblesPath
        path_BVP_label_all = os.path.join(list_path, name_BVP_label_all)
        open_file = open(path_BVP_label_all, "wb")
        pickle.dump(BVP_label_all, open_file)
        open_file.close()

        name_rPPG_all = "rPPG_all.pkl"
        path_rPPG_all = os.path.join(list_path, name_rPPG_all)
        open_file = open(path_rPPG_all, "wb")
        pickle.dump(rPPG_all, open_file)
        open_file.close()


