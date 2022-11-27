import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.plot_functions import set_style, set_size

sys.path.append("src")

from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.pyplot as plt


def fitPCA(train_dataloader, n_pc):
    pca = PCA(n_components=n_pc)
    # scalar = StandardScaler()
    X_frame = []

    for X, _, idx in train_dataloader:
        X_frame.extend(X)

    X_frame = torch.stack(X_frame)
    X_frame = torch.flatten(X_frame, start_dim=1)

    # X_frame = scalar.fit_transform(X_frame)
    X_frame = pca.fit_transform(X_frame)

    return X_frame, pca


def usePCA(test_dataloader, pca):
    X_frame = []

    for X, _, idx in test_dataloader:
        X_frame.extend(X)

    X_frame = torch.stack(X_frame)
    X_frame = torch.flatten(X_frame, start_dim=1)

    # X_frame = scalar.transform(X_frame)
    X_frame = pca.transform(X_frame)

    return X_frame


def plot_explained_var(pca, n_pc, save_path, title="Explained Variance"):
    pc_value = np.arange(1, n_pc + 1)
    exp_vars = [None] * (n_pc)
    set_style()
    for i in pc_value:
        exp_vars[i - 1] = sum(pca.explained_variance_ratio_[:i])

    plt.figure(figsize=(set_size("project", fraction=0.6)))
    plt.plot(pc_value, exp_vars, "r")
    plt.xlabel("Number of PCs"), plt.ylabel("Explained variance")
    plt.xscale("log")
    plt.grid(True, which="minor")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "explained_var.png"))
    plt.savefig(os.path.join(save_path, "explained_var.eps"))


def make_data_split(df_path, save_path, chosen_categories: list = []):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.read_csv(df_path)
    df = df[["filename", "target", "category"]]
    if chosen_categories:
        df = df[df.target.isin(chosen_categories)]
        for i, cat in enumerate(chosen_categories):
            df.loc[df.target == cat, "target"] = i
    df_train, df_test = split_stratified_into_train_val_test(
        df,
        stratify_colname="target",
        frac_train=0.8,
        frac_val=0,
        frac_test=0.2,
        random_state=41,
    )
    df_train.to_csv(os.path.join(save_path, "train.csv"), index=False)
    df_test.to_csv(os.path.join(save_path, "test.csv"), index=False)


# Following function is a modified version from stackoverflowuser2010:
# https://stackoverflow.com/questions/50781562/stratified-splitting-of-pandas-dataframe-into-training-validation-and-test-set
def split_stratified_into_train_val_test(
    df_input,
    stratify_colname="y",
    frac_train=0.6,
    frac_val=0.15,
    frac_test=0.25,
    random_state=None,
):
    """
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    """

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError(
            "fractions %f, %f, %f do not add up to 1.0"
            % (frac_train, frac_val, frac_test)
        )

    if stratify_colname not in df_input.columns:
        raise ValueError("%s is not a column in the dataframe" % (stratify_colname))

    X = df_input  # Contains all columns.
    y = df_input[
        [stratify_colname]
    ]  # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(
        X, y, stratify=y, test_size=(1.0 - frac_train), random_state=random_state
    )
    if frac_val == 0:
        return df_train, df_temp

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(
        df_temp,
        y_temp,
        stratify=y_temp,
        test_size=relative_frac_test,
        random_state=random_state,
    )

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test
