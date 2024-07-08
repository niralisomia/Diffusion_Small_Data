import numpy as np
from numpy import load
import h5py
import pandas as pd
import torch
import lightning as L

from pytorch_lightning import LightningModule
from PL_DeepSTARR import *
from seq_evals_improved import *

samples_file_path = 'samples.npz'
deepSTARR_data = 'DeepSTARR_data.h5'
oracle = 'oracle_DeepSTARR_DeepSTARR_data.ckpt' 

def load_predictions(samples_file_path, deepSTARR_data, oracle):
    
    #load samples from .npz file
    data = load(samples_file_path)
    samples = []
    lst = data.files
    for item in lst:
        samples.append(data[item])

    #load in data
    with h5py.File(deepSTARR_data, 'r') as f:
        # Access the data for the specific X_test key
        x_test = f['X_test'][()]

    #transpose samples to get shape (41186, 4, 249)
    x_synthetic = np.transpose(samples[0], (0, 2, 1))

    #load model
    ckpt_aug_path = oracle
    deepstarr = PL_DeepSTARR.load_from_checkpoint(ckpt_aug_path).eval()

    #make into tensors
    x_test_tensor = torch.from_numpy(x_test).float()
    x_synthetic_tensor = torch.from_numpy(x_synthetic).float()

    #run model predictions
    y_hat_test = deepstarr(x_test_tensor)
    y_hat_syn = deepstarr(x_synthetic_tensor)

    #returns numpy arrays of oracle predictions
    return y_hat_test.detach().numpy(), y_hat_syn.detach().numpy()