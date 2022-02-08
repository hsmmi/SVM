import numpy as np
from my_io import read_dataset_to_X_and_y
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


sample, label = read_dataset_to_X_and_y(
        'dataset/Dataset1.mat', train_size=1, shuffle=True)[0:2]


def plot_decision_boundary_LinearSVC(
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
        xx,
        yy,
        Z,
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'report/{title}.png')


C = [1, 100]

for c in C:
    clf = LinearSVC(C=c, loss='hinge', random_state=1, max_iter=100000)
    clf.fit(sample, label.ravel())
    plot_decision_boundary_LinearSVC(
        clf, sample, label, f'LinearSVC with c = {c}')
    print(f'Accuracy in LinearSVC with c = {c} is',
          f'{np.round(clf.score(sample, label)*100, 2)}%')
