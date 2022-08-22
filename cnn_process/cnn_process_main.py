import numpy as np
import torch
import random
import wandb
# internal modules
from cnn_process import model_pipeline

def cnn_process_main(config):
    # Ensure deterministic behavior
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)


    # Weights & Biases login
    wandb.login()


    # Build, train and analyze the model with the pipeline
    model = model_pipeline.model_pipeline(config)
    return model
