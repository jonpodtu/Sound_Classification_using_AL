import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from active_learning.sampler import sample_uncertainty
from mpl_toolkits.axes_grid1 import ImageGrid
import os
from torch.utils.data import DataLoader
from utils.model_tools import inference_grid
from hydra.utils import to_absolute_path
from datasets import ESC50, Iris
import torch
from sklearn.decomposition import PCA
import matplotlib.patheffects as PathEffects
import pickle
import librosa
from tqdm import tqdm

# Function "set_size" IS FROM https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width, fraction=1, subplots=(1, 1), height_ratio=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    elif width == "project":
        width_pt = 454.10574
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    fig_height_in = fig_height_in * height_ratio

    return (fig_width_in, fig_height_in)


def set_style():
    sns.set_context("paper")
    sns.set(
        context="paper",
        style="whitegrid",
        font="serif",
        font_scale=1,
        rc={
            "xtick.bottom": True,
            "ytick.left": True,
            "axes.edgecolor": ".15",
            "lines.solid_capstyle": "butt",
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "axes.labelsize": 10,
            "font.size": 10,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        },
    )


def plot_results(filename: str, filter_col: str):
    df = pd.read_csv(filename, index_col=0)

    # Get all last epochs
    df = df[df[filter_col] == "True"]
    df = df.drop(
        columns=["train_loss", "test_loss", "train_acc", "epoch", "pool_size", "n"]
    )

    plt.figure(dpi=600)
    g = sns.pointplot(
        data=df, x="train_size", y="test_acc", error_bar=("ci", 95), hue="ActiveLearn"
    )
    g.grid()
    g.set(
        title="Test accuracies over train size",
        xlabel="Train Size",
        ylabel="Accuracy in %",
    )
    plt.legend(title="Active Learning")


def ceil_prec(a, precision=0):
    return np.round(a + 0.5 * 10 ** (-precision), precision)


def floor_prec(a, precision=0):
    return np.round(a - 0.5 * 10 ** (-precision), precision)


def hip_to_be_square(lb, ub, res):
    dist0 = abs(ub[0] - lb[0])
    dist1 = abs(ub[1] - lb[1])

    diff = abs(dist0 - dist1)

    if dist0 > dist1:
        ub[1] = ub[1] + (diff / 2)
        lb[1] = lb[1] - (diff / 2)
    else:
        ub[0] = ub[0] + (diff / 2)
        lb[0] = lb[0] - (diff / 2)

    grid_res = abs(ub[0] - lb[0]) / res

    return lb, ub, grid_res


def make_grid(df, res=300, round_prec=0):
    X = np.column_stack(
        (np.array(df["Principal Component 1"]), np.array(df["Principal Component 2"]))
    )

    assert X.shape[1] == 2, "We can only work in 2 dimensions"

    lb = floor_prec(X.min(0), round_prec)
    ub = ceil_prec(X.max(0), round_prec)

    # We like to square certain plots as they have changing dimensions
    lb, ub, grid_resolution = hip_to_be_square(lb, ub, res)  # If we want to be square
    ub = ub + grid_resolution
    grid = np.mgrid[tuple(slice(i, j, grid_resolution) for i, j in zip(lb, ub))]

    grid_difference = grid_resolution / 2
    extent = (
        lb[0] - grid_difference,
        ub[0] - grid_difference,
        lb[1] - grid_difference,
        ub[1] - grid_difference,
    )
    imshow_kwargs = dict(origin="lower", extent=extent)

    return grid, imshow_kwargs


def plot_al_method(
    df: pd.DataFrame,
    grid,
    grid_preds,
    imshow_kwargs,
    i,
    plot_text=False,
    AL_method="",
    save_folder="",
    explained_var="",
):
    set_style()
    # Grouping by train, pool and sample
    # "Probabilities": prob_grid.reshape(n1, n2, 50).transpose(1, 0, 2),
    n1, n2 = grid.shape[1:]

    df_dict = dict(tuple(df.groupby("Status")))
    plot_dict = {
        "Grid": grid_preds.reshape(n1, n2).T,
    }

    plt.rcParams["font.size"] = 10
    fig = plt.figure(dpi=600)
    imgrid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 1),
        axes_pad=(0.4, 0.5),
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
    )
    for ax, name in zip(imgrid, plot_dict):
        if AL_method == "VAAL":
            cbar_label = "Discriminator value"
            reverse = True
        else:
            cbar_label = "Entropy"
            reverse = True

        im = ax.imshow(
            plot_dict[name],
            cmap=sns.cubehelix_palette(
                start=0.1, rot=-0.5, as_cmap=True, reverse=reverse
            ),
            **imshow_kwargs,
        )
        ax.cax.colorbar(im, label=cbar_label)
        # We make indvidual plots for more control
        custom_lines = []
        custom_labels = []
        for l in df.Labels.unique():
            colors = sns.color_palette("tab10")[1:]
            lw = 1.5
            sns.scatterplot(
                data=df_dict["Pool"][df_dict["Pool"].Labels == l],
                x="Principal Component 1",
                y="Principal Component 2",
                ax=ax,
                fc="none",
                edgecolor=colors[l],
                linewidth=lw,
            )
            sns.scatterplot(
                data=df_dict["Train"][df_dict["Train"].Labels == l],
                x="Principal Component 1",
                y="Principal Component 2",
                ax=ax,
                marker="o",
                color="k",
                edgecolor=colors[l],
                linewidth=lw,
            )
            sns.scatterplot(
                data=df_dict["To sample"][df_dict["To sample"].Labels == l],
                x="Principal Component 1",
                y="Principal Component 2",
                ax=ax,
                marker="P",
                color="k",
                s=100,
                edgecolor=colors[l],
                linewidth=lw,
            )

            custom_lines += [
                Line2D(
                    [0],
                    [0],
                    linestyle="",
                    color="w",
                    marker="s",
                    markerfacecolor=colors[l],
                    markersize=6,
                    markeredgecolor="none",
                )
            ]
        if explained_var:
            ax.set_title(
                f"{AL_method} sampling spectrograms described\nin 2 PCs. Explained variance: {explained_var}"
            )
        else:
            ax.set_title(f"{AL_method} sampling")
        # add legend
        custom_lines += [
            Line2D(
                [0],
                [0],
                linestyle="",
                color="w",
                marker="o",
                markerfacecolor="k",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                color="w",
                markerfacecolor="w",
                fillstyle="none",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                linestyle="",
                color="w",
                marker="P",
                markerfacecolor="k",
                markersize=6,
                markeredgewidth=1,
            ),
        ]

        custom_labels += [
            "Rooster",
            "Footsteps",
            "Chainsaw",
            "Train: {}".format(len(df_dict["Train"])),
            "Pool: {}".format(len(df_dict["Pool"])),
            "To sample: {}".format(len(df_dict["To sample"])),
        ]
        ax.legend(
            custom_lines, custom_labels, loc="best", ncol=2, framealpha=0.5,
        )

        if plot_text:
            for group in ["Train", "To sample"]:
                df_text = df_dict[group]
                for i in range(df_text.shape[0]):
                    txt = ax.text(
                        x=df_text["Principal Component 1"].iloc[i],
                        y=df_text["Principal Component 2"].iloc[i] + 0.03,
                        s=np.round(df_text["AL Values"].iloc[i], 3),
                        fontdict=dict(color="black", size=8),
                    )
                    txt.set_path_effects(
                        [PathEffects.withStroke(linewidth=0.7, foreground="w")]
                    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    df.to_csv(os.path.join(save_folder, f"df.csv"))
    with open(os.path.join(save_folder, f"grid.npy"), "wb") as f:
        np.save(f, grid)
    with open(os.path.join(save_folder, f"grid_preds.npy"), "wb") as f:
        np.save(f, grid_preds)
    with open(os.path.join(save_folder, f"imshow_kwargs.pkl"), "wb") as f:
        pickle.dump(imshow_kwargs, f)
    ax.grid(False)
    ax.tick_params(left=False, bottom=False)
    plt.savefig(os.path.join(save_folder, f"plot.png"), bbox_inches="tight")
    plt.savefig(os.path.join(save_folder, f"plot.eps"), bbox_inches="tight")


def plot_simple(
    device,
    cfg,
    model,
    current_indices,
    pool_indices,
    queries,
    indices,
    save_path,
    preds,
    i,
):
    save_folder = os.path.join(save_path, f"plots/{i}")
    df = make_dataframe(cfg, current_indices, pool_indices, queries)
    if cfg["Simple"].DR == 2:
        ###############
        # Making Grid #
        ###############
        grid, imshow_kwargs = make_grid(df)
        grid_data = grid.transpose(1, 2, 0).reshape(
            -1, 2
        )  # Shape grid into the same as data
        dataloader_grid = DataLoader(grid_data, batch_size=len(grid_data))
        prob_grid = inference_grid(model, device, dataloader_grid)
        grid_preds = sample_uncertainty(prob_grid, indices, n_samples=0)
        grid_preds = grid_preds.numpy()
        plot_al_method(
            df,
            grid,
            grid_preds,
            imshow_kwargs,
            i,
            plot_text=False,
            save_folder=save_folder,
            AL_method="Uncertainty",
        )

    else:
        return
        df["AL Measure"] = [None] * len(df)
        df.loc[indices, "AL Measure"] = list(preds.numpy())
        vmin, vmax = df["AL Measure"].min(), df["AL Measure"].max()
        df["AL Measure"].loc[df["Status"] == "Train"] = vmax

        sns.set_theme()
        ax = sns.relplot(
            data=df,
            x="Principal Component 1",
            y="Principal Component 2",
            hue="Labels",
            size="AL Measure",
            style="Status",
            sizes=(50, 200),
            style_order=["Train", "Pool", "To sample"],
            markers=["o", "$\\bigcirc$", "*"],
            edgecolor=None,
            alpha=0.8,
            palette=sns.color_palette("tab10")[1:],
        )
        legend = ax._legend
        legend.get_texts()[1].set_text("Pig")
        legend.get_texts()[2].set_text("Chirpin Birds")
        legend.get_texts()[3].set_text("Footsteps")
        legend.get_texts()[4].set_text("\nAL Measure")
        legend.get_texts()[10].set_text("\nStatus")
        ax.fig.set_figwidth(12)
        ax.fig.set_figheight(9)
        plt.title("Iteration: {}, Train size: {}".format(i, len(current_indices)))

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, f"{i}.png"), bbox_inches="tight")
        plt.savefig(os.path.join(save_folder, f"{i}.eps"), bbox_inches="tight")


def make_dataframe(cfg, current_indices, pool_indices, queries):
    if cfg.dataset_folder.split("/")[-1] == "Iris":
        train_dataset = Iris(data=to_absolute_path(cfg.paths.train), features=[1, 3])
    else:
        train_dataset = ESC50(
            annotations_file=to_absolute_path(cfg.paths.train),
            audio_dir=to_absolute_path(cfg["Simple"].train),
            DR=cfg["Simple"].DR,
        )
    dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))
    # Format data to be compatible with original plotting ALFunctions
    data = next(iter(dataloader))[0].numpy()
    feature_1 = data[:, 0]
    feature_2 = data[:, 1]
    labels = next(iter(dataloader))[1].numpy()
    df_dict = {
        "Principal Component 1": feature_1,
        "Principal Component 2": feature_2,
        "Labels": labels,
    }
    df = pd.DataFrame.from_dict(df_dict)
    df["Status"] = "Unknown"
    df.iloc[current_indices, -1] = "Train"
    df.iloc[pool_indices, -1] = "Pool"
    df.iloc[queries, -1] = "To sample"
    return df


def plot_vaal(
    discriminator, pool_info, train_info, queries, i, save_path, train_dataset
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        preds = torch.hstack((pool_info[2], train_info[2]))
        mus = torch.vstack((torch.vstack(pool_info[1]), torch.vstack(train_info[1])))
        ids = torch.vstack((torch.vstack(pool_info[0]), torch.vstack(train_info[0])))

        pca = PCA(n_components=2)
        mu_pca = pca.fit_transform(mus.detach().cpu().numpy())
        explained_var = round(pca.explained_variance_ratio_.sum() * 100, 2)
        print(
            f"Performed PCA where the explained variance for 2 componets is: {explained_var}%"
        )

        feature_1 = mu_pca[:, 0]
        feature_2 = mu_pca[:, 1]
        df_dict = {
            "Principal Component 1": feature_1,
            "Principal Component 2": feature_2,
            "AL Values": preds * -1,
            "idx": ids.flatten(),
        }
        df = pd.DataFrame.from_dict(df_dict)
        df["Labels"] = "Unknown"

        dataloader = DataLoader(train_dataset, batch_size=1)
        # Format data to be compatible with original plotting ALFunctions
        for _, label, file_idx in dataloader:
            df.Labels[df.idx == int(file_idx)] = int(label)

        df["Status"] = "Unknown"
        df.Status[df.idx.isin(torch.vstack(train_info[0]).flatten().tolist())] = "Train"
        df.Status[df.idx.isin(torch.vstack(pool_info[0]).flatten().tolist())] = "Pool"
        df.Status[df.idx.isin(queries)] = "To sample"

        grid, imshow_kwargs = make_grid(df, res=600, round_prec=1)
        grid_data = grid.transpose(1, 2, 0).reshape(
            -1, 2
        )  # Shape grid into the same as data

        # Inverse PCA of grid:
        grid_data_inverse = torch.tensor(pca.inverse_transform(grid_data)).float()
        grid_data_inverse = grid_data_inverse.to(device)
        grid_preds = discriminator(grid_data_inverse).cpu().detach().numpy()

    save_folder = os.path.join(save_path, f"plots/{i}")
    plot_al_method(
        df,
        grid,
        grid_preds,
        imshow_kwargs,
        i,
        plot_text=True,
        AL_method="VAAL",
        save_folder=save_folder,
        explained_var=explained_var,
    )


def plot_in_pc(
    device,
    AL_method,
    pool,
    train,
    queries,
    i,
    save_path,
    train_dataset,
    model=None,
    vae=None,
    discriminator=None,
):
    save_folder = os.path.join(save_path, f"plots/pc_{i}")

    # Make dataframe from PC1 and PC2
    dataloader = DataLoader(train_dataset, batch_size=1)
    pca = PCA(n_components=2)
    X_frame = []
    labels = []
    indices = []
    AL_value = []
    for X, y, idx in dataloader:
        X_frame.extend(X)
        labels.extend(y)
        indices.extend(idx)
        X = X.to(device)
        if vae and discriminator:
            with torch.no_grad():
                _, _, mu_temp, _ = vae(X)
                value = discriminator(mu_temp)
        else:
            prob = model(X)
            value = sample_uncertainty(prob, [], n_samples=0)
        AL_value.extend(value.cpu().detach().numpy())

    X_frame = torch.stack(X_frame)
    labels = np.stack(labels)
    indices = np.stack(indices)
    AL_value = np.stack(AL_value).flatten()
    X_frame = torch.flatten(X_frame, start_dim=1)

    data = pca.fit_transform(X_frame)

    feature_1 = data[:, 0]
    feature_2 = data[:, 1]

    df_dict = {
        "Principal Component 1": feature_1,
        "Principal Component 2": feature_2,
        "Labels": labels,
        "AL_Value": AL_value,
    }
    df = pd.DataFrame(df_dict, index=indices)

    df["Status"] = "Unknown"
    df.iloc[train, -1] = "Train"
    df.iloc[pool, -1] = "Pool"
    df.iloc[queries, -1] = "To sample"

    # Make grid based on dataframe
    grid, imshow_kwargs = make_grid(df, res=200)
    grid_data = grid.transpose(1, 2, 0).reshape(
        -1, 2
    )  # Shape grid into the same as data

    # Load PCA model 128x128 and do inverse transformation of grid
    # with open(
    #    to_absolute_path("data/processed/ESC50_3Class/PCA_128x128/pca_model.pkl"), "rb"
    # ) as pickle_file:
    #    pca = pickle.load(pickle_file)
    # FIT PCA ON TRAIN DATA

    explained_var = round(pca.explained_variance_ratio_.sum() * 100, 2)
    print(
        f"Performed PCA where the explained variance for 2 componets is: {explained_var}%"
    )

    # Run grid through model, i.e. VAE (get mu) or Simple (uncertainty values)
    grid_preds = np.empty(grid_data.shape[0])
    for i in tqdm(range(0, grid_data.shape[0]), desc="Making grid"):
        grid_data_invers = pca.inverse_transform(grid_data[i, :])
        grid_data_invers = grid_data_invers.reshape((1, 128, 128))

        if vae and discriminator:
            grid_data_inverse = torch.tensor(grid_data_invers).float()
            grid_data_inverse = grid_data_inverse.to(device)
            with torch.no_grad():
                _, _, mu, _ = vae(grid_data_inverse)
                grid_pred = discriminator(mu).cpu().detach().numpy()
        else:
            dataloader_grid = DataLoader(grid_data_invers, batch_size=1)
            prob_grid = inference_grid(model, device, dataloader_grid)
            grid_pred = sample_uncertainty(prob_grid, indices, n_samples=0)
            grid_pred = grid_pred.numpy()

        grid_preds[i] = grid_pred

    plot_al_method(
        df,
        grid,
        grid_preds,
        imshow_kwargs,
        i,
        plot_text=False,
        AL_method=AL_method,
        save_folder=save_folder,
        explained_var=explained_var,
    )


def plot_spectrogram(
    spec, title=None, ylabel="Frequency bin", aspect="auto", xmax=None, to_db=False
):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram [dB]")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("Frame")
    if to_db:
        spec = librosa.power_to_db(spec)
    im = axs.imshow(spec, origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def combine_plots(AL_method, explained_var=None, plot_text=False, save_folder="./"):
    read_path = to_absolute_path(
        "outputs/2022-11-09/14-40-58/0_run/Uncertainty/Simple/plots/"
    )
    save_folder = read_path

    set_style()
    grid = np.load(os.path.join(read_path, "0/grid.npy"))
    n1, n2 = grid.shape[1:]

    plot_dict = {}
    for i in range(4):
        grid_preds = np.load(os.path.join(*[read_path, str(i), "grid_preds.npy"]))
        plot_dict[f"Grid{i}"] = {
            "grid_pred": grid_preds.reshape(n1, n2).T,
            "df": pd.read_csv(
                os.path.join(*[read_path, str(i), "df.csv"]), index_col=0
            ),
        }

    with open(os.path.join(read_path, "0/imshow_kwargs.pkl"), "rb") as f:
        imshow_kwargs = pickle.load(f)

    plt.rcParams["font.size"] = 10
    fig = plt.figure()  # figsize=(set_size("project", fraction=0.7))
    plt.suptitle(f"{AL_method} sampling on Iris data")
    imgrid = ImageGrid(
        fig,
        111,
        nrows_ncols=(2, 2),
        axes_pad=(0.3, 0.4),
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
    )
    for ax, name in zip(imgrid, plot_dict):
        df = plot_dict[name]["df"]
        df_dict = dict(tuple(df.groupby("Status")))
        if AL_method == "VAAL":
            cbar_label = "Discriminator value"
            reverse = True
        else:
            cbar_label = "Entropy"
            reverse = True

        im = ax.imshow(
            plot_dict[name]["grid_pred"],
            cmap=sns.cubehelix_palette(
                start=0.1, rot=-0.5, as_cmap=True, reverse=reverse
            ),
            **imshow_kwargs,
        )
        ax.cax.colorbar(im, label=cbar_label)
        # We make indvidual plots for more control
        custom_lines = []
        custom_labels = []
        for l in df.Labels.unique():
            colors = sns.color_palette("tab10")[1:]
            lw = 1
            sns.scatterplot(
                data=df_dict["Pool"][df_dict["Pool"].Labels == l],
                x="Sepal width",
                y="Petal width",
                ax=ax,
                fc="none",
                edgecolor=colors[l],
                linewidth=lw,
            )
            sns.scatterplot(
                data=df_dict["Train"][df_dict["Train"].Labels == l],
                x="Sepal width",
                y="Petal width",
                ax=ax,
                marker="o",
                color="k",
                edgecolor=colors[l],
                linewidth=lw,
            )
            sns.scatterplot(
                data=df_dict["To sample"][df_dict["To sample"].Labels == l],
                x="Sepal width",
                y="Petal width",
                ax=ax,
                marker="P",
                color="k",
                s=100,
                edgecolor=colors[l],
                linewidth=lw,
            )

            custom_lines += [
                Line2D(
                    [0],
                    [0],
                    linestyle="",
                    color="w",
                    marker="s",
                    markerfacecolor=colors[l],
                    markersize=6,
                    markeredgecolor="none",
                )
            ]
        if AL_method == "VAAL":
            ax.set_title(
                "VAAL sampling where the latent space is described\nin 2 PCs. Explained variance: {}".format(
                    explained_var
                )
            )
        else:
            train_size = len(df_dict["Train"])
            ax.set_title(f"Train set size {train_size}")
        # add legend
        custom_lines += [
            Line2D(
                [0],
                [0],
                linestyle="",
                color="gray",
                marker="o",
                markerfacecolor="k",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                color="gray",
                markerfacecolor="w",
                fillstyle="none",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                linestyle="",
                color="gray",
                marker="P",
                markerfacecolor="w",
                markersize=6,
                markeredgewidth=1,
            ),
        ]

        custom_labels += [
            "Versicolor",
            "Virginica",
            "Setosa",
            "Train",
            "Pool",
            "To sample",
        ]

        if plot_text:
            for group in ["Train", "To sample"]:
                df_text = df_dict[group]
                for i in range(df_text.shape[0]):
                    txt = ax.text(
                        x=df_text["Sepal width"].iloc[i],
                        y=df_text["Petal width"].iloc[i] + 0.03,
                        s=np.round(df_text["AL Values"].iloc[i], 3),
                        fontdict=dict(color="black", size=8),
                    )
                    txt.set_path_effects(
                        [PathEffects.withStroke(linewidth=0.7, foreground="w")]
                    )
        ax.grid(False)

    fig.legend(
        custom_lines,
        custom_labels,
        loc=(0.04, -0.01),
        ncol=6,
        framealpha=0.5,
        frameon=False,
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(os.path.join(save_folder, f"plot_all.png"), bbox_inches="tight")
    plt.savefig(os.path.join(save_folder, f"plot_all.eps"), bbox_inches="tight")
