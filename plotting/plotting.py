
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from mlxtend.plotting import plot_decision_regions
import graphviz

from .data import load_iris_dataset


def plot_iris_dataset():

    dataset = datasets.load_iris()

    n = min(10, len(dataset.feature_names))
    feature_indices = list(range(n))
    columns=np.asarray(dataset.feature_names)[feature_indices]
    df = pd.DataFrame(dataset.data[:, feature_indices], columns=columns)

    def get_target_name(target):
        name = dataset.target_names[target]
        return name

    df['target'] = dataset.target
    df['species'] = [dataset.target_names[target] for target in dataset.target]

    g = sns.pairplot(df, vars=columns,
                     hue='species', plot_kws={'lw':0, 'alpha':0.7},
                     diag_kws={'bins':25, 'alpha': 0.7})
    g.fig.set_size_inches(7, 7)

    # Make PairPlot lower diagonal
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_visible(False)

    # Modify legend title
    g.fig.get_children()[-1].set_title('Species')


def plot_max_depth_decision_regions():
    X, y = load_iris_dataset()
    X = X[:, :2]
    fig, axarr = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for idx, ax in enumerate(axarr.flat):
        max_depth = idx + 1
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(X, y)
        plot_decision_regions(X, y, model, colors='C0,C1,C2', markers='ooo', res=0.01,
                              hide_spines=False, legend=False, ax=ax)
        ax.set_title(f'max_depth = {max_depth}')
    fig.text(0.5, 0.04, 'Sepal length (cm)', ha='center', va='center')
    fig.text(0.06, 0.5, 'Sepal width (cm)', ha='center', va='center', rotation='vertical')
    plt.show()


def plot_decision_tree(model):

    iris = datasets.load_iris()
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=iris.feature_names[:2],
                               class_names=iris.target_names,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    return graph
