import numpy as np
from my_io import read_dataset_to_X_and_y
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from plot_decision_boundary_SVM import plot_decision_boundary_SVM

sample_train, label_train, sample_test, label_test = read_dataset_to_X_and_y(
        'dataset/Dataset2.mat', train_size=0.8, shuffle=True)
label_train = label_train.ravel()
label_test = label_test.ravel()
sample, label = read_dataset_to_X_and_y(
        'dataset/Dataset2.mat', train_size=1, shuffle=True)[0:2]
label = label.ravel()

test_value = np.array([0.01, 0.04, 0.1, 0.4, 1, 4, 10, 40])

c_gamma = np.array([(c, gamma) for c in test_value for gamma in test_value])


def bulletpoint1():
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=None)

    parameters = {'kernel': ['rbf'], 'C': test_value, 'gamma': test_value}

    clf = GridSearchCV(
        SVC(), parameters, scoring='accuracy', cv=rkf, return_train_score=True)
    clf.fit(sample, label.ravel())
    best_clf = clf.best_estimator_
    best_c, best_gamma = best_clf.C, best_clf.gamma
    best_score = clf.best_score_
    print(f'best C is {best_c} and gamma is {best_gamma}',
          f'with accuracy {np.round(best_score*100, 2)}%')

    clf = SVC(C=best_c, kernel='rbf', gamma=best_gamma, random_state=1)
    clf.fit(sample, label.ravel())
    plot_decision_boundary_SVM(
        clf, sample, label,
        f'Kernel SVM with c = {best_c} and gamma = {best_gamma}')
    test_accuracy = best_clf.score(sample_test, label_test)
    print('Test accuracy with best C ang gamma is',
          f'{np.round(test_accuracy*100, 2)}%')
    print('pause')


# bulletpoint1()


def bulletpoint2():
    train_score = []
    train_var = []
    test_score = []
    test_var = []
    x_axis = []
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=None)
    for c, gamma in c_gamma:
        parameters = {'kernel': ['rbf'], 'C': [c], 'gamma': [gamma]}
        clf = GridSearchCV(
            SVC(), parameters, scoring='accuracy',
            cv=rkf, return_train_score=True)
        clf.fit(sample, label.ravel())
        best_clf = clf.best_estimator_
        train_score.append(best_clf.score(sample_train, label_train))
        train_var.append(
            np.mean((label_train - np.mean(clf.predict(sample_train)))**2))
        test_score.append(best_clf.score(sample_test, label_test))
        test_var.append(
            np.mean((label_test - np.mean(clf.predict(sample_test)))**2))
        x_axis.append(f'C = {c}, gamma = {gamma}')

    plt.plot(x_axis, train_score, label='train_score')
    plt.plot(x_axis, train_var, label='train_var')
    plt.plot(x_axis, test_score, label='test_score')
    plt.plot(x_axis, test_var, label='test_var')
    plt.xticks(rotation=90)
    plt.legend()
    plt.title('Dataset2')
    plt.show()


# bulletpoint2()
