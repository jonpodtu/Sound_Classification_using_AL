from sre_constants import SUCCESS
from models.linear import simple_NN
from utils.train_model import train_model
from utils.PCA import principal_component_analysis, make_plot_pca
import torch.nn as nn
import torchaudio
from torch.utils.data import random_split
from datasets import ESC50
import torch

mode = 'spectrograms' # can also be 'spectrograms'
n_mels = 128
sample_rate = 16000
n_mfcc = 80
top_db = 80
log_mels = False
principal_components = 20

if mode == 'mfcc':
    transformation = torch.nn.Sequential(torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, log_mels=log_mels)) 
    num_features = n_mfcc*1103

if mode == 'spectrograms':
    transformation = torch.nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels = n_mels),torchaudio.transforms.AmplitudeToDB(top_db=top_db))
    num_features = n_mels*1103

data_ESC50 = ESC50(
    annotations_file="data/ESC-50-master/meta/esc50.csv",
    audio_dir="data/ESC-50-master/audio",
    normalize=False,
    transform=transformation
)

num_classes = 50
N = len(data_ESC50)
net = simple_NN(num_features, num_classes)

train_set, val_set = random_split(data_ESC50, [int(N * 0.8), int(N * 0.2)])

make_plot_pca(train_set, len(train_set))

x_train, x_valid = principal_component_analysis(train_set, val_set, principal_components)

x_train = torch.tensor(x_train)
x_valid = torch.tensor(x_valid)


targets_train = [None]*len(train_set)
targets_valid = [None]*len(val_set)

for i in range(len(train_set)):
    targets_train[i] = train_set[i][1]
for i in range (len(val_set)):
    targets_valid[i] = val_set[i][1]

targets_train = torch.tensor(targets_train)
targets_valid = torch.tensor(targets_valid)

#plotmech.plot_PCA_2D(x_train, targets_train)
#plotmech.plot_PCA_2D_v2(x_train, targets_train)

train_model(model=net, training_data=x_train, targets_training=targets_train, validation_data=x_valid, targets_validation=targets_valid, batch_size=16, num_epochs=400, lr=1e-5, loss_function=nn.CrossEntropyLoss())
print(SUCCESS)