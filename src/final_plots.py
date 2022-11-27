import pandas as pd
import utils.plot_functions as pf
import seaborn as sns
import matplotlib.pyplot as plt
from utils.helper import result_handler
import os
from utils.plot_functions import set_style, set_size
import pathlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def find_csv(search_folder):
    files = []

    for (dirpath, dirnames, filenames) in os.walk(search_folder):
        for f in filenames:
            if (pathlib.Path(f).suffix == ".csv") and (f != "df.csv"):
                files.append(os.path.join(dirpath, f))

    return files


def concat_results(paths_to_dfs: list, save_path: str = None):
    df = None
    for path in paths_to_dfs:
        # Apparantly pandas prefer speed over precision:
        # https://stackoverflow.com/questions/47368296/pandas-read-csv-file-with-float-values-results-in-weird-rounding-and-decimal-dig
        new_df = pd.read_csv(path, index_col=0, float_precision="high")
        df = result_handler(df, new_df)

    if save_path:
        df.to_csv(os.path.join(save_path, "results.csv"))
    return df


def make_latex(save_path, df):
    df_mean = df.groupby(["ActiveLearn", "train_size"]).mean()["test_acc"]
    df_mean = df_mean.round(3)
    df_sem = df.groupby(["ActiveLearn", "train_size"]).sem()["test_acc"] * 2
    df_sem = df_sem.round(3)
    df_stats = pd.concat([df_mean, df_sem], axis=1)
    df_stats.columns.values[0] = "Mean"
    df_stats.columns.values[1] = "CI 95"
    df_stats.index.names = ["Sampling", "Train Size"]
    df_stats = df_stats.rename(index={"Uncertain+Random": "Unc. & ran."})
    with open(save_path, "w") as tf:
        tf.write(df_stats.to_latex())


def add_last_and_best(df, save_path: str = False):
    # Ensure last_model is assigned due to earlier errors
    df["last_model"] = False * len(df)
    for i in range(1, len(df)):
        if df.loc[i, "epoch"] == 1:
            df.loc[i - 1, "last_model"] = True
    df.loc[len(df) - 1, "last_model"] = True

    # Look at best model. Note that the next sampled are using the last model
    df["best_model"] = False * len(df)
    intervals = [-1] + df.index[df["last_model"] == True].tolist()
    for i in range(len(intervals) - 1):
        start = intervals[i] + 1
        end = intervals[i + 1] + 1
        df.best_model[df.test_acc[start:end].idxmax()] = True
    if save_path:
        df.to_csv(os.path.join(save_path, "results.csv"))
    return df


#########
# N PCs #
#########
df = pd.read_csv("docs/results/nPCs/n_PCs_new_14619210/results.csv", index_col=0)
df = df[df.last_model == "True"]
set_style()
fig, ax = plt.subplots(figsize=(set_size("project", fraction=0.6)))
sns.lineplot(x="n", y="test_acc", data=df, ax=ax)
sns.scatterplot(x="n", y="test_acc", data=df, ax=ax)
ax.set(
    xlabel="Number of PC's",
    ylabel="Test accuracy in %",
    xticks=[2, 200, 400, 700, 1000, 1300, 1600],
)

axins = inset_axes(ax, "55%", "40%", loc="lower right", borderpad=4)
sns.lineplot(x="n", y="test_acc", data=df[df.n <= 50], ax=axins)
sns.scatterplot(x="n", y="test_acc", data=df[df.n <= 50], ax=axins)
axins.set(xlabel="", ylabel="", xticks=[2, 10, 20, 30, 40, 50])
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--", alpha=0.5)

plt.grid(True)
plt.tight_layout()
plt.savefig("docs/results/nPCs/results.png")
plt.savefig("docs/results/nPCs/results.eps")


############
# SETTINGS #
############
models = ["Simple"]  # , "MobileNet"
set_style()
linewidth = 0.7
order = ["Random", "Uncertain+Random", "Uncertainty", "VAAL"]
al_labels = [
    "Random",
    "Uncertainty & Random",
    "Uncertainty",
    "VAAL",
    "Full dataset (1600)",
]


for model in models:
    save_path = os.path.join("docs/results", model)

    full = find_csv(os.path.join(save_path, "1600"))
    full_df = concat_results(full, save_path)

    dfs = find_csv(os.path.join(save_path, "runs"))

    #########
    # Plots #
    #########
    for mode in ["last_model", "best_model"]:
        df = concat_results(dfs, False)
        df = add_last_and_best(df, save_path)
        df = df[df[mode] == True]

        plt.figure(figsize=(set_size("project", fraction=1)))
        g = sns.pointplot(
            data=df,
            x="train_size",
            y="test_acc",
            errorbar=("se", 2),  # Noneparametric
            hue="ActiveLearn",
            scale=linewidth,
            dodge=0.2,
            hue_order=order,
            errwidth=linewidth * 3,
            capsize=0.1,
        )
        g.set(
            # title="Test accuracies over train size",
            xlabel="Size of training dataset",
            ylabel="Test accuracy in %",
        )

        full_df = add_last_and_best(full_df)
        full = full_df[full_df[mode] == True]
        full_mean = full.test_acc.mean()
        g.axhline(full_mean, label="Full dataset", dashes=[2, 2], color="black")
        handles, labels = g.get_legend_handles_labels()
        plt.grid(True)
        plt.legend(handles=handles, labels=al_labels, title="Sampling Method")
        plt.tight_layout()
        plt.title("Baseline model evaluated on the ESC-50 dataset")
        plt.savefig(os.path.join(save_path, "results_" + mode + ".png"))
        plt.savefig(os.path.join(save_path, "results_" + mode + ".eps"))
        plt.clf()

        full["ActiveLearn"] = "Full"
        make_latex(
            os.path.join(save_path, f"simple_result_{mode}.tex"),
            pd.concat([full, df], axis=0),
        )

###############
# Start Value #
###############
save_path = os.path.join("docs/results", "Simple")
full = [
    os.path.join(dirpath, f)
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(save_path, "1600"))
    for f in filenames
]

full_df = concat_results(full, save_path)
full = full_df[full_df["last_model"] == True]
full_mean = full.test_acc.mean()
print(full_mean)

for al_method in ["uncertainty", "uncertainty+random"]:
    save_path = "docs/results/start_size_test/"
    dfs = [
        os.path.join(dirpath, f)
        for (dirpath, dirnames, filenames) in os.walk(
            os.path.join(save_path, al_method)
        )
        for f in filenames
    ]

    df = pd.read_csv(os.path.join(save_path, "simple_random.csv"), index_col=0)
    df["Start"] = "Random"

    for path in dfs:
        new_df = pd.read_csv(path, index_col=0)
        new_df["Start"] = os.path.basename(path)[:3]
        df = result_handler(df, new_df)

    plt.figure(figsize=(set_size("project", fraction=1)))
    df = df[df["last_model"] == True]
    sns.set_palette(sns.color_palette("YlGnBu_r"))
    g = sns.pointplot(
        data=df,
        x="train_size",
        y="test_acc",
        hue="Start",
        errorbar=("se", 2),  # Noneparametric
        scale=linewidth,
        dodge=0.3,
        errwidth=linewidth * 3,
        capsize=0.1,
    )
    g.axhline(full_mean, label="Full dataset (1600)", dashes=[2, 2], color="black")
    g.set(
        title=f"Different start size using {al_method} sampling",
        xlabel="Size of training dataset",
        ylabel="Test accuracy in %",
        ylim=(11.0, 35.0),
    )
    handles, labels = g.get_legend_handles_labels()
    plt.grid(True)
    plt.legend(handles=handles, labels=labels)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"results_{al_method}.png"))
    plt.savefig(os.path.join(save_path, f"results_{al_method}.eps"))
    full["ActiveLearn"] = "Full"
    make_latex(
        os.path.join(save_path, f"start_{al_method}.tex"), pd.concat([full, df], axis=0)
    )

