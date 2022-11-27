import os
import pandas as pd
import torchaudio
import torch
import numpy as np
import torchvision

from torch.utils.data import Dataset
import torch.nn.functional as F


class ToyData(Dataset):
    def __init__(self, data):
        self.data = pd.read_csv(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
        features = torch.tensor(self.data.iloc[index, :2])
        label = self.data.iloc[index, 2]  # The category label is in the 2nd col
        return features, label, index  # Has to be return as x, y, index

    def get_targets(self):
        # Return all labels
        return self.data.iloc[:, 2]


class Iris(Dataset):
    def __init__(self, data, features=[0, 1, 2, 3]):
        self.data = pd.read_csv(data)
        class_column = [-1]
        self.data = self.data.iloc[:, features + class_column]
        self.features = features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
        features = torch.tensor(self.data.iloc[index, :-1])
        features = features.to(torch.float)
        label = self.data.iloc[index, -1]  # The category label is in the 2nd col
        return features, label, index  # Has to be return as x, y, index

    def get_targets(self):
        # Return all labels
        return self.data.iloc[:, -1]


class ESC50_process(Dataset):
    def __init__(
        self,
        annotations_file,
        audio_dir,
        audio_conf={},
        normalize=True,
        transform=True,
        target_transform=False,
        noisy=False,
        DR=False,  # Dimensionality reduction
        NCs=20,  # No. components
        save_processed=None,  # "data/processed/ESC50/sound",
    ):
        self.audio_labels = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.audio_conf = audio_conf
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = (
            normalize  # Only set to false when creating the mean and std statistics.
        )
        self.noisy = noisy
        self.save_processed = save_processed

        self.DR = DR
        if DR:  # Should be a path to the ..._pca.pt file
            self.components = torch.load(audio_dir).float()
            self.NCs = NCs

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        audio_path = os.path.join(
            self.audio_dir, self.audio_labels.iloc[index, 0]
        )  # Filenames are in the 0 col

        if self.DR:
            out = self.components[index, : self.NCs]

        else:
            waveform, samplerate = torchaudio.load(audio_path)
            # waveform = waveform.to(torch.float)
            filterbank = self.waveform_to_filterbank(
                waveform, samplerate, self.audio_conf.get("n_mels")
            )

            if self.transform:
                resizer = torchvision.transforms.Resize(
                    (
                        self.audio_conf.get("target_length"),
                        self.audio_conf.get("n_mels"),
                    )
                )
                # Add dummy dimensions:
                filterbank = filterbank[None, None, :]
                resized_filterbank = resizer(filterbank)

                # Remove dummy dimensions:
                filterbank = resized_filterbank[0][0]

            if self.normalize:
                mu = self.audio_conf.get("dataset_mean")
                sigma = self.audio_conf.get("dataset_std")
                sigma = torch.where(
                    sigma <= torch.finfo(torch.float32).eps, 1, sigma
                )  # Problem with bin 3 always being the same value
                filterbank = (filterbank - mu) / (sigma)

            if self.noisy:
                filterbank = (
                    filterbank
                    + torch.rand(filterbank.shape[0], filterbank.shape[1])
                    * np.random.rand()
                    / 10
                )  # Add a tensor of noise
                filterbank = torch.roll(
                    filterbank, np.random.randint(-10, 10), 0
                )  # Roll a random interval (+/- 10) along the zero axis

            out = filterbank
        if self.save_processed:
            no_fextensin = os.path.splitext(self.audio_labels.iloc[index, 0])[0]
            filename = no_fextensin + ".pt"
            audio_path = os.path.join(self.save_processed, filename)
            torch.save(out, audio_path)

        # waveform = waveform.to(torch.float)
        label = self.audio_labels.iloc[index, 1]  # The category label is in the 2nd col

        if self.target_transform:
            label = self.target_transform(label)

        return out, label, index  # Has to be return as x, y, index

    def get_targets(self):
        # Return all labels
        return self.audio_labels.iloc[:, 2]

    def waveform_to_filterbank(self, wav, sr, n_mels):
        # Subtract mean
        wav = wav - wav.mean()

        # Calculate filterbank
        filterbank = torchaudio.compliance.kaldi.fbank(
            wav,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=n_mels,
            dither=0.0,
            frame_shift=10,
        )
        return filterbank


"""
        elif self.load_processed:
            no_fextensin = os.path.splitext(self.audio_labels.iloc[index, 0])[0]
            filename = no_fextensin + '.pt'
            audio_path = os.path.join(
                self.load_processed, filename
            )
            out = torch.load(audio_path)
"""


class ESC50(Dataset):
    def __init__(
        self,
        annotations_file,
        audio_dir,
        DR=False,  # Dimensionality reduction
        target_transform=False,
    ):
        self.audio_labels = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.target_transform = target_transform

        self.DR = DR
        if DR:  # Should be a path to the ..._pca.pt file
            self.components = torch.load(audio_dir).float()

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        no_fexten = os.path.splitext(self.audio_labels.iloc[index, 0])[0]
        filename = no_fexten + ".pt"
        audio_path = os.path.join(
            self.audio_dir, filename
        )  # Filenames are in the 0 col

        if self.DR:
            out = self.components[index, : self.DR]

        else:
            out = torch.load(audio_path).float()

        if self.target_transform:
            label = self.target_transform(label)

        # waveform = waveform.to(torch.float)
        label = self.audio_labels.iloc[index, 1]  # The category label is in the 2nd col

        return out, label, index  # Has to be return as x, y, index


################################################
#### Slightly changed dataloader for ESC-US ####
################################################
import os
import pandas as pd
import torchaudio
import torch
import numpy as np
import torchvision

from torch.utils.data import Dataset
import torch.nn.functional as F


class ESC_US_process(Dataset):
    def __init__(
        self,
        annotations_file,
        audio_dir,
        audio_conf={},
        normalize=True,
        transform=True,
        target_transform=False,
        noisy=False,
        save_processed=None,  # "data/processed/ESC50/sound",
    ):
        self.audio_labels = annotations_file
        self.audio_dir = audio_dir
        self.audio_conf = audio_conf
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = (
            normalize  # Only set to false when creating the mean and std statistics.
        )
        self.noisy = noisy
        self.save_processed = save_processed

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        audio_path = os.path.join(
            self.audio_dir, self.audio_labels.iloc[index]
        )  # Filenames are in the 0 col

        try:
            waveform, samplerate = torchaudio.load(audio_path)
        except:
            print("Skiped {} due to reading problems".format(audio_path))
            return False, None

        # waveform = waveform.to(torch.float)
        filterbank = self.waveform_to_filterbank(
            waveform, samplerate, self.audio_conf.get("n_mels")
        )

        if self.transform:
            resizer = torchvision.transforms.Resize(
                (self.audio_conf.get("target_length"), self.audio_conf.get("n_mels"),)
            )
            # Add dummy dimensions:
            filterbank = filterbank[None, None, :]
            resized_filterbank = resizer(filterbank)

            # Remove dummy dimensions:
            filterbank = resized_filterbank[0][0]

        if self.normalize:
            mu = self.audio_conf.get("dataset_mean")
            sigma = self.audio_conf.get("dataset_std")
            sigma = torch.where(
                sigma <= torch.finfo(torch.float32).eps, 1, sigma
            )  # Problem with bin 3 always being the same value
            filterbank = (filterbank - mu) / (sigma)

        if self.noisy:
            filterbank = (
                filterbank
                + torch.rand(filterbank.shape[0], filterbank.shape[1])
                * np.random.rand()
                / 10
            )  # Add a tensor of noise
            filterbank = torch.roll(
                filterbank, np.random.randint(-10, 10), 0
            )  # Roll a random interval (+/- 10) along the zero axis

        out = filterbank
        if self.save_processed:
            no_fextensin = os.path.splitext(
                os.path.basename(self.audio_labels.iloc[index])
            )[0]
            filename = no_fextensin + ".pt"
            audio_path = os.path.join(self.save_processed, filename)
            torch.save(out, audio_path)

        # waveform = waveform.to(torch.float)

        if self.target_transform:
            label = self.target_transform(label)

        return out, index  # Has to be return as x,  index

    def waveform_to_filterbank(self, wav, sr, n_mels):
        # Subtract mean
        wav = wav - wav.mean()

        # Calculate filterbank
        filterbank = torchaudio.compliance.kaldi.fbank(
            wav,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=n_mels,
            dither=0.0,
            frame_shift=10,
        )
        return filterbank


class ESC_US_Normalize(Dataset):
    def __init__(
        self,
        annotations_file,
        audio_dir,
        audio_conf={},
        normalize=True,
        save_processed=None,
    ):
        self.audio_labels = annotations_file
        self.audio_dir = audio_dir
        self.audio_conf = audio_conf
        self.normalize = (
            normalize  # Only set to false when creating the mean and std statistics.
        )
        self.save_processed = save_processed

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        no_fexten = os.path.splitext(self.audio_labels.iloc[index])[0]
        filename = no_fexten + ".pt"
        audio_path = os.path.join(
            self.audio_dir, filename
        )  # Filenames are in the 0 col
        filterbank = torch.load(audio_path).float()

        if self.normalize:
            mu = self.audio_conf.get("dataset_mean")
            sigma = self.audio_conf.get("dataset_std")
            sigma = torch.where(
                sigma <= torch.finfo(torch.float32).eps, 1, sigma
            )  # Problem with bin 3 always being the same value
            filterbank = (filterbank - mu) / (sigma)

        out = filterbank
        if self.save_processed:
            no_fextensin = os.path.splitext(
                os.path.basename(self.audio_labels.iloc[index])
            )[0]
            filename = no_fextensin + ".pt"
            audio_path = os.path.join(self.save_processed, filename)
            torch.save(out, audio_path)

        return out, index  # Has to be return as x, y, index


class ESC_US(Dataset):
    def __init__(
        self, annotations_file, audio_dir,
    ):
        self.audio_labels = annotations_file
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
        no_fextensin = os.path.splitext(self.audio_labels.iloc[index][0])[0]
        filename = no_fextensin + ".pt"
        audio_path = os.path.join(
            self.audio_dir, filename
        )  # Filenames are in the 0 col
        out = torch.load(audio_path).float()

        return out  # Has to be return as x, y, index

