
from sklearn import datasets
import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve
from mlxtend.plotting import plot_decision_regions
import graphviz

FONTSIZE = 20


def plot_2D_iris(fontsize=None):
    if fontsize is None:
        fontsize = FONTSIZE
    X, y = datasets.load_iris(return_X_y=True)
    X = X[:, :2]

    labels = ['setosa', 'versicolor', 'virginica']
    fig, ax = plt.subplots(figsize=(10, 8))
    for target, label in zip(range(3), labels):
        ax.scatter(X[y == target, 0], X[y == target, 1],
                   color=f'C{target}', s=100,
                   label=label, lw=0)
    ax.set_xlabel('Sepal length (cm)', fontsize=fontsize)
    ax.set_ylabel('Sepal width (cm)', fontsize=fontsize)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    leg = plt.legend(title='Iris species', fontsize=18)
    plt.setp(leg.get_title(), fontsize=fontsize)
    plt.show()


def plot_tree_decision_regions(clf: DecisionTreeClassifier, fontsize=None):

    if fontsize is None:
        fontsize = FONTSIZE

    X, y = datasets.load_iris(return_X_y=True)
    X = X[:, :2]

    labels = ['setosa', 'versicolor', 'virginica']
    fig, ax = plt.subplots(figsize=(10, 8))
    with plt.style.context({'lines.markersize': 10}):
        plot_decision_regions(X, y, clf, colors='C0,C1,C2', markers='ooo',
                              hide_spines=False, ax=ax)
    ax.set_xlabel('Sepal length (cm)', fontsize=fontsize)
    ax.set_ylabel('Sepal width (cm)', fontsize=fontsize)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    leg = plt.legend(title='Iris species', fontsize=18)
    for idx, label in enumerate(labels):
        leg.get_texts()[idx].set_text(label)
    plt.setp(leg.get_title(), fontsize=fontsize)
    plt.show()


def plot_decision_tree(model: DecisionTreeClassifier):

    iris = datasets.load_iris()
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=iris.feature_names[:2],
                               class_names=iris.target_names,
                               impurity=False,
                               filled=True,
                               rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    return graph


def plot_classification_vs_regression(fontsize=None):

    if fontsize is None:
        fontsize = FONTSIZE

    np.random.seed(2)

    markersize = 50

    X_r = np.linspace(0, 1, 100)
    y_r_true = jv(X_r, 2)
    y_r = np.random.normal(y_r_true, 0.02)

    fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    ax_reg = axarr[1]
    ax_reg.scatter(X_r, y_r, s=markersize, alpha=0.7, lw=0, color='C2')
    ax_reg.plot(X_r, y_r_true, color='C2')
    ax_reg.set_xlabel('x', fontsize=fontsize)
    ax_reg.set_ylabel('y', fontsize=fontsize)
    ax_reg.set_title('Regression', fontsize=fontsize)
    ax_reg.xaxis.set_ticklabels([])
    ax_reg.yaxis.set_ticklabels([])

    X_c, y_c = datasets.make_blobs(n_samples=500, n_features=2,
                                   centers=2, random_state=2)

    ax_c = axarr[0]
    for label in range(2):
        label_mask = y_c == label
        ax_c.scatter(X_c[label_mask, 0], X_c[label_mask, 1], s=markersize,
                     c=f'C{label}', alpha=0.7, label=label)
    ax_c.set_xlabel('$X_1$', fontsize=fontsize)
    ax_c.set_ylabel('$X_2$', fontsize=fontsize)
    ax_c.set_title('Classification', fontsize=fontsize)
    ax_c.xaxis.set_ticklabels([])
    ax_c.yaxis.set_ticklabels([])

    clf = LogisticRegression(random_state=2).fit(X_c, y_c)

    xmin, xmax = X_c[:, 0].min(), X_c[:, 0].max()
    coef = clf.coef_
    intercept = clf.intercept_

    def line(x0):
        return (-(x0 * coef[0, 0]) - intercept[0]) / coef[0, 1]

    ax_c.plot([xmin, xmax], [line(xmin), line(xmax)], ls='--', color='k')

    leg = ax_c.legend(title='Class labels', fontsize=18)
    plt.setp(leg.get_title(), fontsize=fontsize)
    plt.show()


def plot_data_representation():

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.axis('equal')

    # Draw features matrix
    ax.vlines(range(6), ymin=0, ymax=9, lw=1)
    ax.hlines(range(10), xmin=0, xmax=5, lw=1)
    font_prop = dict(size=18, family='monospace')
    ax.text(-1, -1, "Feature Matrix ($X$)", size=20)
    ax.text(0.1, -0.3, r'n_features $\longrightarrow$', **font_prop)
    ax.text(-0.1, 0.1, r'$\longleftarrow$ n_samples', rotation=90,
            va='top', ha='right', **font_prop)

    # Draw labels vector
    ax.vlines(range(8, 10), ymin=0, ymax=9, lw=1)
    ax.hlines(range(10), xmin=8, xmax=9, lw=1)
    ax.text(7, -1, "Target Vector ($y$)", size=20)
    ax.text(7.9, 0.1, r'$\longleftarrow$ n_samples', rotation=90,
            va='top', ha='right', **font_prop)

    ax.set_ylim(10, -2)


def plot_validation_curve():

    x = np.linspace(0, 1, 1000)
    y1 = -(x - 0.5) ** 2
    y2 = y1 - 0.33 + np.exp(x - 1)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(x, y2, lw=10, alpha=0.5, color='C0')
    ax.plot(x, y1, lw=10, alpha=0.5, color='C1')

    ax.text(0.15, 0.2, "training score", rotation=45, size=16, color='C0')
    ax.text(0.2, -0.05, "validation score", rotation=20, size=16, color='C1')

    ax.text(0.02, 0.1, r'$\longleftarrow$ High Bias', size=18,
            rotation=90, va='center')
    ax.text(0.98, 0.1, r'$\longleftarrow$ High Variance $\longrightarrow$',
            size=18, rotation=90, ha='right', va='center')
    ax.text(0.48, -0.12, 'Best$\\longrightarrow$\nModel', size=18,
            rotation=90, va='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 0.5)

    ax.set_xlabel(r'model complexity $\longrightarrow$', size=14)
    ax.set_ylabel(r'model score $\longrightarrow$', size=14)

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())

    ax.set_title("Validation Curve Schematic", size=16)
    plt.show()


def plot_max_depth_validation(clf, X, y, n_jobs=2, fontsize=None):

    if fontsize is None:
        fontsize = FONTSIZE

    max_depths = list(range(1, 10))
    train_scores, val_scores = validation_curve(clf, X, y,
                                                param_name='max_depth',
                                                param_range=max_depths,
                                                scoring='accuracy',
                                                cv=10,
                                                n_jobs=n_jobs)
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(max_depths, train_mean)
    plt.fill_between(max_depths,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.2, label='Training')
    plt.plot(max_depths, val_mean)
    plt.fill_between(max_depths,
                     val_mean + val_std,
                     val_mean - val_std,
                     alpha=0.2, label='Validation')
    plt.xlabel('max_depth', fontsize=fontsize)
    plt.ylabel('Accuracy', fontsize=fontsize)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.show()
