
from sklearn import datasets


def load_iris_dataset():

    X, y = datasets.load_iris(return_X_y=True)

    return X, y
