from sklearn.datasets import make_blobs, make_classification
import matplotlib.pyplot as plt

def main():
    # method taken from https://www.projectpro.io/recipes/create-simulated-data-for-clustering-in-python#mcetoc_1g2u5r5t97s
    features_1, clusters_1 = make_blobs(n_samples = 2000, n_features = 2, centers = 3, cluster_std = 2, shuffle = False, center_box=(-20,20), random_state=42)
    plt.scatter(features_1[:,0], features_1[:,1], c=clusters_1)
    plt.show()

if __name__ == "__main__":
    main()