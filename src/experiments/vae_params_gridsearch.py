import sys
sys.path.append("../../src")
from tqdm import tqdm
import torch.optim as optim
from datasets import ESC50
from active_learning.vaal.model import VAE
from models.simple import simple_NN
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from vae_tools import train_vae, train_epoch_vae, valid_epoch_vae
import os

def train_model(vae, device, train_dataloader, test_dataloader, num_epochs, n_features, n_classes, num_hidden, lr, wd):
    # We are simply doing holdout 
        
    # Initialize model for each run:
    model = simple_NN(num_features=n_features, num_output=n_classes, num_hidden = num_hidden)
    print("Model Used: Neural Network")
        
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=wd
    )

    train_len = len(train_dataloader.sampler)
    test_len = len(test_dataloader.sampler)

    for epoch in tqdm(range(num_epochs)):
        # Inspiration from https://gist.github.com/eugeniaring/1c60a806d6ecb2b7e1eefc5d89ed7958#file-py
        train_loss, train_correct = train_epoch_vae(
            vae, model, device, train_dataloader, optimizer
        )

        train_loss = train_loss / train_len
        train_acc = train_correct / train_len

        if epoch+1 %10 == 0:
            print("Train Loss: {tl}, Train Acc: {ta}".format(tl = train_loss, ta = train_acc))


    test_loss, test_correct = valid_epoch_vae(vae, model, device, test_dataloader)

    test_loss = test_loss / test_len
    test_acc = test_correct / test_len * 100

    print("Test Loss: {tel}, Test Acc: {tea}".format(tel = test_loss, tea = test_acc))

    return test_acc

def main():
    dataset_size = 1600
    train_epochs = 300
    train_epochs_vae = 150
    batch_size = 32
    dataset_dimension = 64
    learning_rate_model = 0.001 # Change
    wd_model = 0.001 # Change

    betas = [0.01, 0.5, 1.0, 2.0]
    batch_sizes_vae = [32, 64, 128, 256] 
    learning_rates = [5e-6, 1e-5, 5e-4, 1e-4]
    #betas = [0.01, 0.5]
    #batch_sizes_vae =[32, 64]
    #learning_rates = [5e-6, 1e-5]

    train_set = ESC50(
        annotations_file=to_absolute_path("data/processed/ESC50/train.csv"),
        audio_dir=to_absolute_path("data/processed/ESC50/train_{n}x{n}".format(n=dataset_dimension)),
        DR=False,
    )

    test_set = ESC50(
        annotations_file=to_absolute_path("data/processed/ESC50/test.csv"),
        audio_dir=to_absolute_path("data/processed/ESC50/test_{n}x{n}".format(n=dataset_dimension)),
        DR=False,
    )

    train_dataloader_model = DataLoader(
        train_set,
        batch_size = batch_size,
        drop_last = False,
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size = 400,
        drop_last = False, 
    )

    savefolder = 'outputs/vae_gridsearch'
    exists=os.path.exists(savefolder)
    if not exists:
        os.makedirs(savefolder)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hyper_parameters_results = np.zeros(
        (len(learning_rates), len(betas), len(batch_sizes_vae))
    )
    hyperparameter_dataframe = pd.DataFrame(columns = ['LR', 'Beta', 'Batch Size', 'VAE Loss', 'Test Acc'])
    for p1, lr in enumerate(learning_rates):
        for p2, beta in enumerate(betas):
            for p3, batch_size_vae in enumerate(batch_sizes_vae):
                train_dataloader_vae = DataLoader(
                    train_set,
                    batch_size = batch_size_vae,
                    drop_last = False,
                )
                vae = VAE(32)
                vae, _, vae_loss = train_vae(
                    vae, 
                    device, 
                    train_dataloader_vae, 
                    dataset_size, 
                    train_epochs_vae, 
                    batch_size_vae, 
                    beta,
                    lr,
                    savefolder = savefolder,
                    plot = False,
                    )

                test_acc = train_model(
                    vae=vae,
                    device=device, 
                    train_dataloader=train_dataloader_model,
                    test_dataloader=test_dataloader, 
                    num_epochs = train_epochs, 
                    n_features = 32, 
                    n_classes = 50,
                    num_hidden=[256,128],
                    lr = learning_rate_model,
                    wd = wd_model,
                    )

                hyper_parameters_results[p1, p2, p3] = test_acc
                print("LR: {lr}, BETA: {beta}, BATCH: {batch}, Acc: {acc}".format(lr = lr, beta = beta, batch = batch_size_vae, acc = test_acc))
                hyperparameter_dataframe = hyperparameter_dataframe.append({'LR' : lr, 'Beta' : beta, 'Batch Size' : batch_size_vae, 'VAE Loss' : vae_loss, 'Test Acc' : test_acc}, ignore_index = True)
    
    hyperparameter_dataframe.to_pickle(savefolder+'/'+'hyperparameter_df.pkl')

    best_acc = np.max(hyper_parameters_results)
    lr_idx, beta_idx, bs_idx = np.where(hyper_parameters_results == best_acc)
    #best_lr = learning_rates[lr_idx.item()]
    #best_beta = betas[beta_idx.item()]
    #best_batch = batch_sizes_vae[bs_idx.item()]


    print("BEST LR: ", lr_idx)
    print("BEST BETA: ", beta_idx)
    print("BEST BATCH: ", bs_idx)
    print("BEST ACC: ", best_acc)

if __name__ == '__main__':
    main()