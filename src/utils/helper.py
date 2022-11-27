from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.metrics import pairwise_distances

# from dtw import dtw
import librosa
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from scipy.io import wavfile
import scipy.spatial.distance as dist


def result_handler(old_results, results):
    if not isinstance(old_results, pd.DataFrame):
        return results

    results = pd.concat([old_results, results], ignore_index=True)

    return results


def get_filenames(data_path: str) -> list:
    """Gives path to folder with wav file which should be predicted.

    Keyword arguments:
    data_path -- path to folder with wav files

    Returns:
    file_list -- list of filenames
    prefix -- the prefix of the path the files where collected from

    TODO: Fileextension check
    """
    file_list = os.listdir(data_path)
    prefix = os.path.abspath(data_path)
    file_list = [
        os.path.join(file_name) for file_name in file_list if file_name.endswith(".wav")
    ]
    return file_list
