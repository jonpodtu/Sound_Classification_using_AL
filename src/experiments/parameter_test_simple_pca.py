import sys
sys.path.append("../../src")
from tqdm import tqdm
import torch.optim as optim
from datasets import ESC50
from models.simple import simple_NN
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import torch
import numpy as np
from experiment_plottools import plot_results
import scipy.stats as st
import pandas as pd
import os

def train_model(device, n_folds, train_dataset, test_dataloader, batch_size,dataset_size, num_epochs, n_features, n_classes, num_hidden, savefolder):
    kfold = KFold(n_splits=n_folds, shuffle=True)
    # Use entire dataset
    current_indices = np.arange(dataset_size)
    
    # Define lr and wd to get best model parameters for, given the model setup
    learning_rates = [0.01, 0.001, 0.0005]
    weight_decays = [0.01,  0.005, 0.001, 0.0005, 0.0001] 
    
    # Log all hp
    hyper_parameters_results = np.zeros(
        (len(learning_rates), len(weight_decays), n_folds)
    )

    # Five folds to find parameters:
    for fold, (train_ids, val_ids) in tqdm(
        enumerate(kfold.split(current_indices)), desc="Performing Cross-validation"
    ):
        # Loop over lr
        for p1, lr in enumerate(learning_rates):
            # Loop over wd
            for p2, wd in enumerate(weight_decays):
                # Define dataloaders for the inner loop
                train_sampler = SubsetRandomSampler(current_indices[train_ids])
                val_sampler = SubsetRandomSampler(current_indices[val_ids])
                train_dataloader = DataLoader(
                    train_dataset,
                    sampler=train_sampler,
                    batch_size=batch_size,
                    drop_last=True,
                )
                val_dataloader = DataLoader(
                    train_dataset,
                    sampler=val_sampler,
                    batch_size=batch_size,
                    drop_last=False,
                )

                train_len = len(train_dataloader.sampler)
                val_len = len(val_dataloader.sampler)
                
                # Initialize model for each run:
                model = simple_NN(num_features=n_features, num_output=n_classes, num_hidden=num_hidden)
                model = model.to(device)
                optimizer = optim.Adam(
                    model.parameters(), lr=lr, weight_decay=wd
                )

                # Train each model in inner loop  
                for epoch in range(num_epochs):
                    # Inspiration from https://gist.github.com/eugeniaring/1c60a806d6ecb2b7e1eefc5d89ed7958#file-py
                    train_loss, train_correct = train_epoch(
                        model, device, train_dataloader, optimizer
                    )
                    valid_loss, valid_correct = valid_epoch(
                        model, device, val_dataloader
                    )

                    train_loss = train_loss / train_len
                    valid_loss = valid_loss / val_len
                    train_acc = train_correct / train_len
                    valid_acc = valid_correct / val_len

                    if epoch+1 %10 == 0:
                        print("Train Loss: {tl}, Train Acc: {ta}, Val Loss: {vl}, Val Acc: {va}".format(tl = train_loss, ta = train_acc, vl = valid_loss, va = valid_acc))
                # Log matrix of hp:
                hyper_parameters_results[p1, p2, fold] = valid_acc

    # Calculate best hyperparameters
    hyper_parameters_mean = np.mean(hyper_parameters_results, axis=2)
    best_acc = np.max(hyper_parameters_mean)
    lr_idx, wd_idx = np.where(hyper_parameters_mean == best_acc)

    #########################################
    ##### Train and test of final model #####
    #########################################
    # Final model on full set (All test data)
    # Test on entire dataset:
    test_losses = np.empty(5)
    test_corrects = np.empty(5)
    test_acc_finals = np.empty(5)
    loss_df = pd.DataFrame(columns = ['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc', 'Fold'])
    for fold in tqdm(range(n_folds), desc="Performing Cross-validation, outer"
    ):
        train_sampler = SubsetRandomSampler(current_indices)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True,
        )

        train_len = len(train_dataloader.sampler)
        val_len = len(val_dataloader.sampler)
        test_len = len(test_dataloader.sampler)

        model = simple_NN(num_features=n_features, num_output=n_classes, num_hidden=num_hidden)
        model = model.to(device)

        optimizer = optim.Adam(
            model.parameters(), lr=learning_rates[lr_idx.item()], weight_decay=weight_decays[wd_idx.item()]
        )

        for epoch in range(num_epochs):
            # Inspiration from https://gist.github.com/eugeniaring/1c60a806d6ecb2b7e1eefc5d89ed7958#file-py
            train_loss, train_correct = train_epoch(
                model, device, train_dataloader, optimizer
            )

            train_loss = train_loss / train_len
            train_acc = train_correct / train_len

            # Calculate test_acc underway for the final model:
            test_loss, test_correct = valid_epoch(model, device, test_dataloader)

            test_loss = test_loss / test_len
            test_acc = test_correct / test_len

            # Save results to dataframe:
            loss_df = loss_df.append({'Epoch' : epoch, 'Train Loss': train_loss, 'Train Acc' : train_acc, 'Test Loss' : test_loss, 'Test Acc' : test_acc, 'Fold' : fold}, ignore_index = True)

            if epoch+1 %10 == 0:
                print("Train Loss: {tl}, Train Acc: {ta}".format(tl = train_loss, ta = train_acc))
        
        test_losses[fold], test_corrects[fold] = valid_epoch(model, device, test_dataloader)
        test_acc_finals[fold] = test_corrects[fold] / test_len

    # Save loss dataframe:
    loss_df.to_pickle(savefolder+'/'+'loss_df_{num_hidden}.pkl'.format(num_hidden = num_hidden[0]))

    print("Fold number {n}, Test Loss Mean, Std: {telm}, {tels}, Test Acc Mean, Std: {team}, {teas}".format(n = fold, telm = np.mean(test_loss), tels = np.std(test_loss), team = np.mean(test_acc_finals), teas = np.std(test_acc_finals)))

    return np.mean(test_acc_finals), np.std(test_acc_finals), best_acc, learning_rates[lr_idx.item()], weight_decays[wd_idx.item()]

def train_epoch(model, device, train_dataloader, optimizer):
    # Forward pass, then backward pass, then update parameters:
    running_loss, correct = 0.0, 0
    loss_fn = torch.nn.CrossEntropyLoss()
    torch.set_grad_enabled(True)

    model.train()
    for X, y, _ in train_dataloader:
        X = X.to(device)
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

def valid_epoch(model, device, val_dataloader):
    model.eval()
    running_loss, correct = 0.0, 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for X, y, _ in val_dataloader:
        X = X.to(device)
        # print(f"Validation: X shape = {X.shape}")
        output = model(X)

        y = y.to(device)
        # Val loss
        loss = loss_fn(output, y)
        running_loss += loss.item() * X.size(0)

        _, predictions = torch.max(output.data, 1)
        correct += (predictions == y).sum().item()

    return running_loss, correct

def main():
    dataset_size = 1600
    train_epochs = 300 # ---------------- CHANGE!!!!!!
    batch_size = 32
    dataset_dimension = 64
    model = 'nn'
    npc = 700

    train_set = ESC50(
        annotations_file=to_absolute_path("data/processed/ESC50/train.csv"),
        audio_dir=to_absolute_path("data/processed/ESC50/PCA_{n}x{n}/train.pt".format(n=dataset_dimension)),
        DR=npc,
    )

    test_set = ESC50(
        annotations_file=to_absolute_path("data/processed/ESC50/test.csv"),
        audio_dir=to_absolute_path("data/processed/ESC50/PCA_{n}x{n}/test.pt".format(n=dataset_dimension)),
        DR=npc,
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size = 400,
        drop_last = False, 
    )

    # Make sure we can save our results from training the final models:

    savefolder = 'outputs/test_params_simple_npc{npc}_{epochs}'.format(npc = npc, epochs = train_epochs)
    exists=os.path.exists(savefolder)
    if not exists:
        os.makedirs(savefolder)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden_combinations = [[16,8], [32,16],[64,32],[128,64],[256,128],[512,256]]
    fold_results = pd.DataFrame(columns = ['LR', 'WD', 'Mean Acc over folds', 'Mean Final Acc', 'Std Final Acc'], index = [str(config) for config in hidden_combinations])
    print("Train Epochs: ", train_epochs, " Batch Size: ", batch_size)
    for num_hidden in hidden_combinations:
        print("Using {first} neurons in first layer and {second} neurons in second layer".format(first = num_hidden[0], second = num_hidden[1]))
        test_acc_mean, test_acc_std, best_acc, lr, wd = train_model(
            device=device, 
            n_folds=5, #Change
            train_dataset=train_set,
            test_dataloader=test_dataloader, 
            batch_size = batch_size,
            dataset_size = dataset_size, 
            num_epochs = train_epochs, 
            n_features = npc, 
            n_classes = 50,
            num_hidden=num_hidden,
            savefolder = savefolder
            )
        fold_results.loc[str(num_hidden)] = [lr, wd, best_acc, test_acc_mean, test_acc_std]
        print("Test Accuracy for {config}: {acc}".format(config=str(num_hidden), acc = test_acc_mean))
    # Save this dataframe:
    fold_results.to_pickle(savefolder+'/'+"fold_results_simple_pca_{npc}.pkl".format(npc = npc))
        
if __name__ == '__main__':
    main()