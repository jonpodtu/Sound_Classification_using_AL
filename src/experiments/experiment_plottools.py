import sys

sys.path.append("../../src")
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plot_functions import set_style, set_size


def plot_results(df, type):
    set_style()
    linewidth = 0.3
    plt.figure(figsize=(set_size("project", fraction=1.0)))
    sns.lineplot(
        data=df,
        x="Epoch",
        y="Train {type}".format(type=type),
        label="Train {type}".format(type=type),
    )
    g = sns.lineplot(
        data=df,
        x="Epoch",
        y="Test {type}".format(type=type),
        label="Test {type}".format(type=type),
    )
    g.set(
        title="{type} for {model_name}".format(
            type=type, model_name="Simple Neural Network"
        ),
        xlabel="Epoch",
        ylabel=type,
    )
    # handles, labels = g.get_legend_handles_labels()
    # plt.legend(handles=handles, labels=labels, title="Active Learning")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig('model_{type}_{model}{suff}.png'.format(type=type, model = model, suff = suffix))
    plt.show()


def plot_results_size(dataframes, lrs, savefolder, filename, save=True):
    set_style()
    linewidth = 0.1
    fig = plt.figure(figsize=(set_size("project", fraction=1.0)))
    for i in range(4):
        ax = fig.add_subplot(2, 2, 1 + i)
        df = dataframes[i]
        g = sns.lineplot(
            data=df,
            x="Training Iteration",
            y="VAE Loss",
            hue="Dimension",
            palette=sns.color_palette(),
            ax=ax,
        )
        handles, labels = g.get_legend_handles_labels()
        ax.set(
            title="Learning Rate: {lr}".format(lr=lrs[i]),
            xlabel=None,
            ylabel=None,
            yscale="log",
        )
        plt.legend([], [], frameon=False)
    fig.supxlabel("Train Iteration", x=0.55)
    fig.supylabel("VAE Loss")
    plt.suptitle("VAE Loss over Spectrogram Dimension", x=0.55)
    fig.legend(
        handles=handles,
        labels=labels,
        loc=(0.1, 0.03),
        ncol=3,
        framealpha=0.5,
        frameon=True,
    )
    # plt.legend(title="Dimension", loc = "upper right")
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(savefolder + "/" + filename + ".png", bbox_inches="tight")
        plt.savefig(savefolder + "/" + filename + ".pdf", bbox_inches="tight")
        plt.savefig(savefolder + "/" + filename + ".eps", bbox_inches="tight")
    plt.show()


def plot_results_both(df, title, savefolder, filename, save=True):
    df["Test Acc"] = df["Test Acc"] * 100.0
    df["Train Acc"] = df["Train Acc"] * 100.0
    set_style()
    linewidth = 0.3
    fig = plt.figure(figsize=(set_size("project", fraction=1.0)))
    label = ["Loss", "Acc"]
    yaxis = ["Loss", "Accuracy in %"]
    for i in range(2):
        ax = fig.add_subplot(1, 2, 1 + i)
        sns.lineplot(
            data=df,
            x="Epoch",
            y="Train {type}".format(type=label[i]),
            label="Train {type}".format(type=label[i]),
        )
        g = sns.lineplot(
            data=df,
            x="Epoch",
            y="Test {type}".format(type=label[i]),
            label="Test {type}".format(type=label[i]),
        )
        handles, labels = g.get_legend_handles_labels()
        ax.set(
            title=None, xlabel=None, ylabel=yaxis[i],
        )
        plt.legend([], [], frameon=False)
    fig.supxlabel("Epoch", x=0.55)
    plt.suptitle(title, x=0.55)
    fig.legend(
        handles=handles,
        labels=["Train", "Test"],
        loc=(0.7, 0.04),
        ncol=2,
        framealpha=0.5,
        frameon=True,
    )
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(savefolder + "/" + filename + ".png", bbox_inches="tight")
        plt.savefig(savefolder + "/" + filename + ".pdf", bbox_inches="tight")
        plt.savefig(savefolder + "/" + filename + ".eps", bbox_inches="tight")
    plt.show()
