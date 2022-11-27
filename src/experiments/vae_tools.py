import sys
sys.path.append("../../src")
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import pandas as pd

def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label, _ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def train_vae(vae, device, train_dataloader, dataset_size, train_epochs, batch_size, beta, lr, savefolder, plot):
    train_iterations = (
        dataset_size * train_epochs
    ) // batch_size
    data = read_data(train_dataloader)

    optim_vae = optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    vae = vae.to(device)

    beta = beta
    losses = np.empty(1)

    for iter_count in tqdm(range(train_iterations)):

        labeled_imgs, labels = next(data)

        labeled_imgs = labeled_imgs.to(device)
        labels = labels.to(device)

        # Removed number of VAE steps.
        recon, z, mu, logvar = vae(labeled_imgs)
        #print("mu: ", mu)
        unsup_loss = vae_loss(x=labeled_imgs, recon=recon, mu=mu, logvar=logvar, beta=beta)

        # print(f"Total VAE loss: {total_vae_loss}")
        optim_vae.zero_grad()
        unsup_loss.backward()
        optim_vae.step()
        
        try:
            if iter_count == 0:
                losses = np.array(
                    [
                        unsup_loss.item(),
                    ]
                )
            else:
                losses = np.vstack(
                    [
                        losses,
                        [
                            unsup_loss.item(),
                        ],
                    ]
                )
        except:
            print("An error occured when documenting lossses for the VAE")


        if(iter_count+1 % 10 == 0):
            print(f"Iteration {iter_count}/{train_iterations}", "     VAE LOSS: {loss}".format(loss=unsup_loss))

    loss_df = pd.DataFrame(
        losses,
        columns=[
            "VAE Loss",
        ],
    )
    loss_df.to_pickle(savefolder+'/'+'vae_loss_b{b}_bs{bs}_lr{lr}.pkl'.format(b=beta, bs = batch_size, lr = lr))
    if plot:
        NotImplementedError
        # Plots loss rate of VAAL

    return vae, loss_df, losses[-1] # Return vae and final loss

def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD

def train_epoch_vae(vae, model, device, train_dataloader, optimizer):
    # Forward pass, then backward pass, then update parameters:
    running_loss, correct = 0.0, 0
    loss_fn = torch.nn.CrossEntropyLoss()
    torch.set_grad_enabled(True)
    
    vae.eval()
    model.train()
    for X, y, _ in train_dataloader:
        with torch.no_grad():
            _,_,X,_ = vae(X.to(device))
        #print(X)
        #print(X.shape)
        y = y.to(device)

        optimizer.zero_grad()
        # Input must be given in form [N, Channels, Width, Height]
        output = model(X)
        # compute gradients given loss # Is this supposed to be none if do not choose so? (Fine tune == False)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        # Train loss
        running_loss += loss.item() * X.size(0)

        # Train accuracy
        _, predictions = torch.max(output.data, 1)
        correct += (predictions == y).sum().item()

    return running_loss, correct

def valid_epoch_vae(vae, model, device, val_dataloader):
    model.eval()
    running_loss, correct = 0.0, 0
    loss_fn = torch.nn.CrossEntropyLoss()
    vae.eval()
    for X, y, _ in val_dataloader:
        with torch.no_grad():
            _,_,X,_ = vae(X.to(device))
        # print(f"Validation: X shape = {X.shape}")
        output = model(X)

        y = y.to(device)
        # Val loss
        loss = loss_fn(output, y)
        running_loss += loss.item() * X.size(0)

        _, predictions = torch.max(output.data, 1)
        correct += (predictions == y).sum().item()

    return running_loss, correct