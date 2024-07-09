import numpy as np
from numpy import load
import h5py
import pandas as pd
import torch
import lightning as L

from pytorch_lightning import LightningModule
from PL_DeepSTARR import *

samples_file_path = 'samples.npz'
deepSTARR_data = 'DeepSTARR_data.h5'
oracle_path = 'oracle_DeepSTARR_DeepSTARR_data.ckpt'

class EmbeddingExtractor:
    def __init__(self):
        self.embedding = None

    def hook(self, module, input, output):
        self.embedding = output.detach()

def extract_data(samples_file_path, deepSTARR_data):
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

    #make into tensors
    x_test_tensor = torch.from_numpy(x_test).float()
    x_synthetic_tensor = torch.from_numpy(x_synthetic).float() 

    return x_test_tensor, x_synthetic_tensor

def load_deepstarr(oracle_path):
    
    #load model
    ckpt_aug_path = oracle_path
    deepstarr = PL_DeepSTARR.load_from_checkpoint(ckpt_aug_path).eval()

    return deepstarr

def load_predictions(x_test_tensor, x_synthetic_tensor, deepstarr):

    #run model predictions
    y_hat_test = deepstarr(x_test_tensor)
    y_hat_syn = deepstarr(x_synthetic_tensor)

    #returns numpy arrays of deepstarr predictions from samples and x test
    return y_hat_test.detach().numpy(), y_hat_syn.detach().numpy()


extractor = EmbeddingExtractor()
def get_penultimate_embeddings(model, x):
    # Find the penultimate layer
    for name, module in model.named_modules():
        if name == 'model.batchnorm6':
            handle = module.register_forward_hook(extractor.hook)
            break
    else:
        raise ValueError("Could not find 'model.batchnorm6' layer")

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    # Remove the hook
    handle.remove()

    return extractor.embedding

