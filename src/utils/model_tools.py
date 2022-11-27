import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import torch.optim as optim
from models.simple import simple_NN
from models.logistic_regression import LogisticRegression
from models.convolutional_networks import efficient_net, mobile_net
import random


def train(
    TaskLearner: str,
    train_dataset,
    current_indices,
    device: str,
    test_dataloader,
    cfg,
    seed,
):
    """
        This training loop is taken more or less directly from the deep learning course
    """
    current_indices = np.asarray(current_indices)

    ########################################
    ##### HYPER PARAMETER OPTIMIZATION #####
    ########################################

    # Hyperparameter settings
    n_folds = cfg.n_folds
    learning_rates = cfg[TaskLearner]["learning_rates"]
    weight_decays = cfg[TaskLearner]["weight_decays"]
    hyper_parameters_reults = np.zeros(
        (len(learning_rates), len(weight_decays), n_folds)
    )

    kfold = KFold(n_splits=n_folds, shuffle=True)

    for fold, (train_ids, val_ids) in tqdm(
        enumerate(kfold.split(current_indices)), desc="Performing Cross-validation"
    ):
        for p1, lr in enumerate(learning_rates):
            for p2, wd in enumerate(weight_decays):
                # Making train and validation splits
                train_sampler = SubsetRandomSampler(current_indices[train_ids])
                val_sampler = SubsetRandomSampler(current_indices[val_ids])
                train_dataloader = DataLoader(
                    train_dataset,
                    sampler=train_sampler,
                    batch_size=cfg.batch_size,
                    drop_last=True,
                )
                val_dataloader = DataLoader(
                    train_dataset,
                    sampler=val_sampler,
                    batch_size=cfg.batch_size,
                    drop_last=False,
                )
                # Reset model
                model, optimizer = initialize_model(
                    TaskLearner,
                    n_features=cfg[TaskLearner].DR,
                    n_classes=cfg.n_class,
                    learning_rate=lr,
                    weight_decay=wd,
                    optim_type=cfg[TaskLearner]["Optimizer"],
                    device=device,
                    seed=seed,
                )

                val_len = len(val_dataloader.sampler)

                # initialize the early_stopping object
                # early stopping patience; how long to wait after last time validation loss improved.
                early_stopping = EarlyStopping(patience=cfg.early_stop, verbose=True)

                for epoch in range(cfg.num_epochs):
                    # Inspiration from https://gist.github.com/eugeniaring/1c60a806d6ecb2b7e1eefc5d89ed7958#file-py
                    train_loss, train_correct = train_epoch(
                        model, device, train_dataloader, optimizer
                    )
                    valid_loss, valid_correct = valid_epoch(
                        model, device, val_dataloader
                    )

                    early_stopping(valid_loss, model)
                    if early_stopping.early_stop:
                        # print("Early stopping")
                        break

                    valid_acc = valid_correct / val_len

                hyper_parameters_reults[p1, p2, fold] = valid_acc

    # Choose best combination of parameters

    hyper_parameters_reults = np.mean(hyper_parameters_reults, axis=2)
    best_acc = np.max(hyper_parameters_reults)
    lr_idx, wd_idx = np.where(hyper_parameters_reults == best_acc)

    # We perform a random choice in case of multiple best accuracies
    if len(lr_idx) > 1:
        choice = random.randrange(0, len(lr_idx))
        lr_idx = lr_idx[choice]
        wd_idx = wd_idx[choice]

    lr = float(learning_rates[int(lr_idx)])
    wd = float(weight_decays[int(wd_idx)])

    print(
        "Chosen hyperparameters based on {} folds: \n LR: {} \n Weight Decay: {} \nwith a mean acc. of {}".format(
            n_folds, lr, wd, best_acc * 100
        )
    )

    #########################################
    ##### Train and test of final model #####
    #########################################
    # Final model on full set (Also validation data)
    train_sampler = SubsetRandomSampler(current_indices)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=cfg.batch_size, drop_last=True,
    )

    # Reset model
    model, optimizer = initialize_model(
        TaskLearner,
        n_features=cfg[TaskLearner].DR,
        n_classes=cfg["n_class"],
        learning_rate=lr,
        weight_decay=wd,
        optim_type=cfg[TaskLearner]["Optimizer"],
        device=device,
        seed=seed,
    )

    train_len = len(train_dataloader.sampler)
    val_len = len(test_dataloader.sampler)

    results = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "epoch": [],
        "lr": [],
        "weight_decay": [],
        "params_acc": [],
        "last_model": [],
        "current_indices": [],
    }

    early_stopping = EarlyStopping(patience=cfg.early_stop, verbose=True)

    for epoch in range(cfg.num_epochs):
        train_loss, train_correct = train_epoch(
            model, device, train_dataloader, optimizer
        )
        valid_loss, valid_correct = valid_epoch(model, device, test_dataloader)

        train_loss = train_loss / train_len
        train_acc = train_correct / train_len * 100
        valid_loss = valid_loss / val_len
        valid_acc = valid_correct / val_len * 100

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(valid_loss)
        results["test_acc"].append(valid_acc)
        results["epoch"].append(epoch + 1)
        results["lr"].append(lr)
        results["weight_decay"].append(wd)
        results["params_acc"].append(best_acc)
        results["current_indices"].append(list(current_indices))

        if (epoch + 1) % 10 == 0:
            print(
                "Epoch:{}/{} Train. Loss:{:.3f} Val. Loss:{:.3f} Training Acc. {:.2f} % Val. Acc. {:.2f} %".format(
                    epoch + 1,
                    cfg.num_epochs,
                    train_loss,
                    valid_loss,
                    train_acc,
                    valid_acc,
                )
            )
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            results["last_model"].append(True)
            break
        elif epoch + 1 == cfg.num_epochs:
            results["last_model"].append(True)
        else:
            results["last_model"].append(False)

    return model, results


def train_epoch(model, device, train_dataloader, optimizer):
    # Forward pass, then backward pass, then update parameters:
    running_loss, correct = 0.0, 0
    loss_fn = torch.nn.CrossEntropyLoss()
    torch.set_grad_enabled(True)

    model.train()
    for X, y, _ in train_dataloader:
        X = X.to(device)
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
        y = y.to(device)
        # print(f"Validation: X shape = {X.shape}")
        output = model(X)

        y = y.to(device)
        # Val loss
        loss = loss_fn(output, y)
        running_loss += loss.item() * X.size(0)

        _, predictions = torch.max(output.data, 1)
        correct += (predictions == y).sum().item()

    return running_loss, correct


def inference(model, device, pool_dataloader):
    model.eval()
    model.to(device)

    all_preds = []
    all_indices = []

    for X, _, indices in pool_dataloader:
        X = X.to(device)
        preds = model(X)
        preds = preds.cpu().data
        all_preds.extend(preds)
        all_indices.extend(indices)

    all_preds = torch.stack(all_preds)
    return all_preds, all_indices


def inference_grid(model, device, pool_dataloader):
    model.eval()
    model.to(device)

    all_preds = []

    for X in pool_dataloader:
        X = X.float()
        X = X.to(device)
        preds = model(X)
        preds = preds.cpu().data
        all_preds.extend(preds)

    all_preds = torch.stack(all_preds)
    return all_preds


def initialize_model(
    model_type,
    n_features,
    n_classes,
    learning_rate=None,
    weight_decay=None,
    optim_type=None,
    device="cpu",
    seed=random.randint(0, 10000),
):
    if model_type == "LogReg":
        model = LogisticRegression(num_features=n_features, num_output=n_classes)

    elif model_type == "Simple":
        assert type(n_features) == int, "Set an int for datareduction (DR)"
        torch.manual_seed(seed)
        model = simple_NN(num_features=n_features, num_output=n_classes)

    elif model_type == "EfficientNet":
        torch.manual_seed(seed)
        model = efficient_net(label_dimension=n_classes, pretrain=True)

    elif model_type == "MobileNet":
        torch.manual_seed(seed)
        model = mobile_net(label_dimension=n_classes, pretrain=True)

    else:
        raise ValueError(
            "The chosen model is not implemented. Choose either 'Simple', or 'CNN'. "
        )

    if optim_type == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optim_type == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    else:
        optimizer = None
        print("No optimizer was chosen. Returning none for optimizer")

    return model.to(device), optimizer

# The whole class is from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        # if self.verbose:
        # self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
