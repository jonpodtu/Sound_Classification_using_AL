from sre_constants import SUCCESS
from models.cnn import ConvolutionalNN
from utils.train_model import train_model
from utils.PCA import principal_component_analysis
import torch.nn as nn
import torchaudio
import torchvision
from torch.utils.data import random_split
from datasets import ESC50
import torch
import numpy as np


mode = 'mfcc' # can also be 'spectrograms'
n_mels = 128
sample_rate = 1600
n_mfcc = 80
top_db = 80
log_mels = False
principal_components = 40

if mode == 'mfcc':
    transformation = torch.nn.Sequential(torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, log_mels=log_mels), 
                                        torchvision.transforms.Resize((224,224)))
    num_features = n_mfcc*1103

if mode == 'spectrograms':
    transformation = torch.nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels = n_mels),
                                        torchaudio.transforms.AmplitudeToDB(top_db=top_db))
    num_features = n_mels*1103

if mode == 'mel-spectrograms':
    transformation = torch.nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels = n_mels),
                                        torchaudio.transforms.AmplitudeToDB(top_db=top_db), 
                                        torchvision.transforms.Resize((224,224)))


data_ESC50 = ESC50(
    annotations_file="data/ESC-50-master/meta/esc50.csv",
    audio_dir="data/ESC-50-master/audio",
    normalize=False,
    transform=transformation
)

num_classes = 50
N = len(data_ESC50)
model_class = ConvolutionalNN(pretrained = True, fine_tune = False, num_classes = num_classes)
model = model_class.build_model()

train_set, val_set = random_split(data_ESC50, [int(N * 0.8), int(N * 0.2)])

x_train = [None]*len(train_set)
targets_train = [None]*len(train_set)
x_valid = [None]*len(val_set)
targets_valid = [None]*len(val_set)

#gray_to_rgb = torchvision.transforms.Lambda(lambda x: torch.stack([xx,xx,xx],2) )

for i in range(len(train_set)):
    x_train[i] = np.concatenate((train_set[i][0].numpy(),train_set[i][0].numpy(),train_set[i][0].numpy()),0)
    targets_train[i] = train_set[i][1]
for i in range (len(val_set)):
    x_valid[i] = np.concatenate((val_set[i][0].numpy(),val_set[i][0].numpy(),val_set[i][0].numpy()),0)
    targets_valid[i] = val_set[i][1]

targets_train = torch.tensor(targets_train)
targets_valid = torch.tensor(targets_valid)
x_train = torch.tensor(x_train)
x_valid = torch.tensor(x_valid)


train_model(model=model, training_data=x_train, targets_training=targets_train, validation_data=x_valid, targets_validation=targets_valid, batch_size=100, num_epochs=200, lr=1e-4, loss_function=nn.CrossEntropyLoss())
print(SUCCESS)

