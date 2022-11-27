from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import utils.dataset_split as df_split

iris = load_iris()

use_classes = [
    0,
    1,
    2,
]  # None (all three classes or a list of class labels to use, e.g., [0, 1])
use_features = [
    0,
    1,
    2,
    3
]  # The two features to use. Should be in the set (0, 1, 2, 3)

X = iris["data"][:, use_features]
y = iris["target"]

plt.figure(1, figsize=(12, 12))
colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]

for k, col in enumerate(colors):
    cluster_data = y == k
    plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=50)

plt.title("2D Toy dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

data = {
    "Feature 1": X[:, 0],
    "Feature 2": X[:, 1],
    "Feature 3": X[:, 2],
    "Feature 4": X[:, 3],
    "Class": y,
}
df = pd.DataFrame(data)
df.to_csv("data/toy_data/Iris.csv", index=False)


def make_data_split(df):
    df_train, df_val, df_test = df_split.split_stratified_into_train_val_test(
        df,
        stratify_colname="Class",
        frac_train=0.6,
        frac_val=0.15,
        frac_test=0.25,
        random_state=None,
    )
    df_train.to_csv("data/processed/toy_data/Iris_train.csv", index=False)
    df_val.to_csv("data/processed/toy_data/Iris_val.csv", index=False)
    df_test.to_csv("data/processed/toy_data/Iris_test.csv", index=False)


make_data_split(df)
