import sys
sys.path.append("../../src")
from tqdm import tqdm
import torch.optim as optim
from datasets import ESC50
from active_learning.vaal.model import VAE, Discriminator
from models.logistic_regression import LogisticRegression
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd
from utils.plot_functions import set_style, set_size
from experiment_plottools import plot_results_size
from vae_tools import train_vae
import os

def main():
    dataset_size = 1600
    train_epochs_vaal = 150
    batch_size = 32
    dataset_dimension = 32
    multipliers = [1, 2, 4]
    beta=1
    lr = 1e-4 # [1e-5, 5e-4, 1e-4, 5e-3]

    loss_dataframe = pd.DataFrame()

    savefolder = 'outputs/test_of_sizes_folds'
    exists=os.path.exists(savefolder)
    if not exists:
        os.makedirs(savefolder)

    for multiplier in multipliers:

        dim = dataset_dimension*multiplier

        train_set = ESC50(
            annotations_file=to_absolute_path("data/processed/ESC50/train.csv"),
            audio_dir=to_absolute_path("data/processed/ESC50/train_{n}x{n}".format(n=dim)),
            DR=False,
        )

        for fold in range(5):
            train_dataloader_full = DataLoader(
                train_set,
                batch_size = batch_size,
                drop_last = False,
            )

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            vae = VAE(z_dim=32, nc=1, multiplier=multiplier)
            vae, loss_vae, _= train_vae(vae, device, train_dataloader_full, dataset_size, train_epochs_vaal, batch_size, beta, lr, savefolder, plot=False)
            dim_col = [dim]*loss_vae.shape[0]
            loss_vae['Dimension'] = dim_col
            loss_vae['N'] = fold
            loss_dataframe = loss_dataframe.append(loss_vae)
    
    loss_dataframe.to_pickle(savefolder+"/"+"lossdf_epochs_{e}_dim_{ld}_bs_{bs}_lr_{lr}.pkl".format(e=train_epochs_vaal, ld=dataset_dimension*multipliers[-1], bs = batch_size, lr=lr))  
    print(loss_dataframe)

    #plot_results_size(loss_dataframe, savefolder, "lossplot_{e}_{ld}_{bs}.pkl".format(e=train_epochs_vaal, ld=dataset_dimension*multipliers[-1], bs = batch_size))

if __name__ == '__main__':
    main()