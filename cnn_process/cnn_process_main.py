import numpy as np
import torch
import random
import wandb
# internal modules
from cnn_process import model_pipeline

def cnn_process_main(config):
    # Ensure deterministic behavior
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


    # Weights & Biases login
    wandb.login()


    # Build, train and analyze the model with the pipeline
    model = model_pipeline.model_pipeline(config)
