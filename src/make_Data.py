import torch
from torch.utils.data import DataLoader
import os
from datasets import ESC50_process, ESC50
from utils.dataset import fitPCA, usePCA, plot_explained_var, make_data_split
import pickle

###########
## Paths ##
###########
audio_dir = "data/raw/ESC-50-master/audio"  # "data/raw/ESC-50-master/audio"
save_folder = "data/processed/ESC50"  # <-- USER INPUT HERE
# 3_class: "data/processed/ESC50_3Class"

############################
#####   Make Splits    #####
############################
meta = (
    "data/raw/ESC-50-master/meta/esc50.csv"  # "data/raw/ESC-50-master/meta/esc50.csv"
)

# 3_class: make_data_split(meta, save_folder, chosen_categories=[2, 14, 25])
make_data_split(meta, save_folder)  # CHANGE THIS

annotation_train = os.path.join(save_folder, "train.csv")
annotation_test = os.path.join(save_folder, "test.csv")

# Settings for mels and target lengths in sets (tuples of (n_mels, target_length)). Example [(128, 498), (40, 40)]
# PC's will be made from the last setting
settings = [(32, 32),(64, 64), (128, 128), (128, 498)]  # <-- USER INPUT HERE

############################
##### Make Spectograms #####
############################
for n_mels, target_length in settings:
    # Find mean and standard deviation
    audio_config = {"n_mels": n_mels, "target_length": target_length}

    train_set = ESC50_process(
        annotations_file=annotation_train,
        audio_dir=audio_dir,
        audio_conf=audio_config,
        transform=True,
        normalize=False,
    )

    train_dataset = DataLoader(train_set, batch_size=len(train_set))

    for spectrogram, label, index in train_dataset:
        audio_config["dataset_mean"] = torch.mean(spectrogram, axis=0)
        audio_config["dataset_std"] = torch.std(spectrogram, axis=0)

    print(
        "Dataset Mean: ",
        audio_config["dataset_mean"],
        "Dataset STD: ",
        audio_config["dataset_std"],
    )

    train_dir = os.path.join(save_folder, f"train_{target_length}x{n_mels}")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    test_dir = os.path.join(save_folder, f"test_{target_length}x{n_mels}")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    torch.save(audio_config["dataset_mean"], os.path.join(train_dir, "mean.pt"))
    torch.save(audio_config["dataset_std"], os.path.join(train_dir, "std.pt"))

    # Output datasets
    train_set = ESC50_process(
        annotations_file=annotation_train,
        audio_dir=audio_dir,
        audio_conf=audio_config,
        transform=True,
        normalize=True,
        save_processed=train_dir,
    )

    test_set = ESC50_process(
        annotations_file=annotation_test,
        audio_dir=audio_dir,
        audio_conf=audio_config,
        transform=True,
        normalize=True,
        save_processed=test_dir,
    )

    train_dataloader = DataLoader(train_set, batch_size=1, drop_last=False)
    test_dataloader = DataLoader(test_set, batch_size=1, drop_last=False)

    for x, y, i in train_dataloader:
        continue

    for x, y, i in test_dataloader:
        continue

###########################################
#####    Make Principal Components    #####
###########################################
train_set = ESC50(
    annotations_file=annotation_train,
    audio_dir=os.path.join(save_folder, f"train_{target_length}x{n_mels}"),
    DR=False,
)

test_set = ESC50(
    annotations_file=annotation_test,
    audio_dir=os.path.join(save_folder, f"test_{target_length}x{n_mels}"),
    DR=False,
)

pca_folder = os.path.join(save_folder, f"PCA_{target_length}x{n_mels}")
if not os.path.exists(pca_folder):
    os.makedirs(pca_folder)

train_dataloader = DataLoader(train_set)
test_dataloader = DataLoader(test_set)

n_pc = len(train_set)

train_new, pca = fitPCA(train_dataloader, n_pc)
test_new = usePCA(test_dataloader, pca)

plot_explained_var(pca, n_pc, pca_folder, title=False)

train_new = torch.from_numpy(train_new)
test_new = torch.from_numpy(test_new)

torch.save(train_new, os.path.join(pca_folder, f"train.pt"))
torch.save(test_new, os.path.join(pca_folder, f"test.pt"))
with open(os.path.join(pca_folder, "pca_model.pkl"), "wb") as pickle_file:
    pickle.dump(pca, pickle_file)
print("Completed construction of dataset")
