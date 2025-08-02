import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D


class PCA:
    def __init__(self, k):
        self.k = k
        self.Uk_ = None
        self.means_ = None

    def fit(self, X):
        self.means_ = np.mean(X, axis=0)
        X_centered = X - self.means_

        S = np.cov(X_centered, rowvar=False)

        eigvals, eigvectors = np.linalg.eig(S)

        idx = np.argsort(eigvals)[::-1]
        sorted_eigvectors = eigvectors[:, idx]
        self.Uk_ = sorted_eigvectors[:, : self.k]

    def transform(self, X):
        X_centered = X - self.means_

        X_pca = X_centered @ self.Uk_

        return np.array(X_pca)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


df = pd.read_csv("./data/PCA.csv", index_col="id")
df.drop(columns="Unnamed: 32", inplace=True)

X = df.drop(columns="diagnosis")
Y = df["diagnosis"]

pca_2d = PCA(k=2)
pca_3d = PCA(k=3)

X_2d = pca_2d.fit_transform(X)
X_3d = pca_3d.fit_transform(X)

colors = {"M": "red", "B": "blue"}
labels = Y.map(colors)

fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.8])
custom_legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Malignant",
        markerfacecolor="red",
        markeredgecolor="k",
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Benign",
        markeredgecolor="k",
        markerfacecolor="blue",
        markersize=10,
    ),
]

ax2d = fig.add_subplot(gs[0])
ax2d.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, alpha=0.6, edgecolors="k", s=60)
ax2d.set_title("PCA with k = 2", fontsize=16, fontweight="bold")
ax2d.set_xlabel("PC1", fontsize=12)
ax2d.set_ylabel("PC2", fontsize=12)
ax2d.grid(True)
ax2d.legend(handles=custom_legend)

ax3d = fig.add_subplot(gs[1], projection="3d")
ax3d.scatter(
    X_3d[:, 0],
    X_3d[:, 1],
    X_3d[:, 2],
    c=labels,
    edgecolors="k",
    alpha=0.6,
    s=60,
)
ax3d.set_title("PCA with k = 3", fontsize=16, fontweight="bold")
ax3d.set_xlabel("PC1", fontsize=12)
ax3d.set_ylabel("PC2", fontsize=12)
ax3d.set_zlabel("PC3", fontsize=12)
ax3d.view_init(elev=15, azim=70)

plt.tight_layout()
plt.show()
