import torch
from torch.utils.data import DataLoader
import os
from datasets import ESC50_process, ESC50, Iris
from utils.dataset import fitPCA, usePCA, plot_explained_var, make_data_split

train_set = Iris(data="/zhome/e3/3/139772/Desktop/Bachelorproject/data/processed/Iris/train_new.csv")
test_set = Iris(data="/zhome/e3/3/139772/Desktop/Bachelorproject/data/processed/Iris/test.csv")

train_dataloader = DataLoader(train_set)
test_dataloader = DataLoader(test_set)

n_pc = 2

train_new, pca = fitPCA(train_dataloader, n_pc)
test_new = usePCA(test_dataloader, pca)

train_new = torch.from_numpy(train_new)
test_new = torch.from_numpy(test_new)

torch.save(train_new, os.path.join("/zhome/e3/3/139772/Desktop/Bachelorproject/data/processed/Iris", "train_pca.pt"))
torch.save(test_new, os.path.join("/zhome/e3/3/139772/Desktop/Bachelorproject/data/processed/Iris", "test_pca.pt"))