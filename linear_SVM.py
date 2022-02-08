import numpy as np
from my_io import read_dataset_to_X_and_y
from sklearn.svm import LinearSVC

from plot_decision_boundary_SVM import plot_decision_boundary_SVM


sample, label = read_dataset_to_X_and_y(
        'dataset/Dataset1.mat', train_size=1, shuffle=True)[0:2]

C = [1, 100]

for c in C:
    clf = LinearSVC(C=c, loss='hinge', random_state=1, max_iter=100000)
    clf.fit(sample, label.ravel())
    plot_decision_boundary_SVM(
        clf, sample, label, f'LinearSVC with c = {c}')
    print(f'Accuracy in LinearSVC with c = {c} is',
          f'{np.round(clf.score(sample, label)*100, 2)}%')
