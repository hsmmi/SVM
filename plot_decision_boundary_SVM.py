from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC


def plot_decision_boundary_SVM(
        clf: LinearSVC, sample: np.ndarray, label: np.ndarray, title: str):
    plt.figure(figsize=(5, 5))

    # obtain the support vectors through the decision function
    decision_function = clf.decision_function(sample)

    support_vector_indices = \
        np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = sample[support_vector_indices]

    plt.scatter(sample[:, 0], sample[:, 1], c=label, s=30, cmap='bwr')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50)
    )
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(
        xx, yy, Z, colors="k", levels=[-1, 0, 1],
        alpha=0.5, linestyles=["--", "-", "--"]
    )
    plt.scatter(
        support_vectors[:, 0], support_vectors[:, 1], s=100,
        linewidth=1, facecolors="none", edgecolors="k"
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'report/{title}.png')
