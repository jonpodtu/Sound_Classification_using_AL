import sys
sys.path.append("../../src")
from tqdm import tqdm
import torch.optim as optim
from datasets import ESC50
from active_learning.vaal.model import VAE
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader

import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import numpy as np
import pandas as pd
from utils.plot_functions import set_style, set_size

def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label, _ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def train_vae(vae, device, train_dataloader, dataset_size, train_epochs, batch_size):
    train_iterations = (
        dataset_size * train_epochs
    ) // batch_size
    data = read_data(train_dataloader)

    optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
    vae.train()
    vae = vae.to(device)

    i = 0
    beta = 1
    losses = np.empty(2)

    for iter_count in tqdm(range(train_iterations)):

        labeled_imgs, labels = next(data)

        labeled_imgs = labeled_imgs.to(device)
        labels = labels.to(device)

        # VAE step

        # Removed number of VAE steps.
        recon, z, mu, logvar = vae(labeled_imgs)
        #print("mu: ", mu)
        unsup_loss = vae_loss(x=labeled_imgs, recon=recon, mu=mu, logvar=logvar, beta=beta)

        # print(f"Total VAE loss: {total_vae_loss}")
        optim_vae.zero_grad()
        unsup_loss.backward()
        optim_vae.step()
        
        try:
            if i == 0:
                losses = np.array(
                    [
                        unsup_loss.item(),
                        iter_count,
                    ]
                )
            else:
                losses = np.vstack(
                    [
                        losses,
                        [
                            unsup_loss.item(),
                            iter_count,
                        ],
                    ]
                )
        except:
            print("An error occured when documenting lossses for the VAE")

        # Save model for every epoch (not iteration)     
        if (i+1) % (dataset_size/batch_size) == 0.0: # One epoch
            save_model(vae, 'outputs/vae_models','vae')
            print("Saved model for given iteration", i)
        # For every tenth epoch, save extra model:
        if (i+1) % ((dataset_size/batch_size)*10) == 0.0:
            NotImplementedError("Save Epoch")
            save_model(vae, 'outputs/vae_models','vae{iteration}'.format(iteration=i))
            print("Saved model for given iteration (tenth epoch)", i)
        
        if(i % 10 == 0):
            print(f"Iteration {i}/{train_iterations}", "     VAE LOSS: {loss}".format(loss=unsup_loss))
        i+=1

    loss_df = pd.DataFrame(
        losses,
        columns=[
            "VAE Loss",
            "Training Iteration",
        ],
    )
    return vae, loss_df

def save_model(vae, savefolder, modelname):
    exists=os.path.exists(savefolder)
    if not exists:
        os.makedirs(savefolder)
    torch.save(
        vae.state_dict(), os.path.join(savefolder, "{name}.pt".format(name=modelname)),
    )


def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    #print("MSE: ", MSE)
    #print("KLD: ", KLD)
    return MSE + KLD

def plot_results(df):
    set_style()
    #linewidth = 0.7
    plt.figure(figsize=(set_size("project", fraction=0.8)))
    g = sns.lineplot(data=df, x="Training Iteration", y="VAE Loss", hue="Dimension", palette = sns.color_palette())
    g.set(
        title="VAE Loss over spectrogram dimension",
        xlabel="Training Iteration",
        ylabel="VAE Loss",
        yscale="log",
    )
    handles, labels = g.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title="Active Learning")
    plt.legend(title="Dimension", loc = "upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('vae_loss_savedmodel.png')
    plt.show()

def main():
    dataset_size = 1600
    train_epochs_vaal = 150
    batch_size = 32
    dataset_dimension = 32
    multipliers = [2]

    loss_dataframe = pd.DataFrame()
    for multiplier in multipliers:

        dim = dataset_dimension*multiplier

        train_set = ESC50(
            annotations_file=to_absolute_path("data/processed/ESC50/train.csv"),
            audio_dir=to_absolute_path("data/processed/ESC50/train_{n}x{n}".format(n=dim)),
            DR=False,
        )

        train_dataloader_full = DataLoader(
            train_set,
            batch_size = batch_size,
            drop_last = False,
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        vae = VAE(z_dim=32, nc=1, multiplier=multiplier)
        vae, loss_vae = train_vae(vae, device, train_dataloader_full, dataset_size, train_epochs_vaal, batch_size)
        dim_col = [dim]*loss_vae.shape[0]
        loss_vae['Dimension'] = dim_col
        loss_dataframe = loss_dataframe.append(loss_vae)
    
    loss_dataframe.to_pickle("loss_df_savedmodel.pkl")  
    print(loss_dataframe)

    plot_results(loss_dataframe)

if __name__ == '__main__':
    main()