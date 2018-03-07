# Supervised machine learning in Python with scikit-learn

[![Build Status](https://travis-ci.org/jrbourbeau/madpy-ml-sklearn-2018.svg?branch=master)](https://travis-ci.org/jrbourbeau/madpy-ml-sklearn-2018)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/jrbourbeau/madpy-ml-sklearn-2018/master?filepath=machine-learning-sklearn-madpy.ipynb)

Materials for "Supervised machine learning in Python with scikit-learn" talk at the Madison Python (MadPy) meetup. A live version of the slides can be found at https://jrbourbeau.github.io/madpy-ml-sklearn-2018/.


## Installation

The dependencies for generating the slides are:

- `numpy`
- `pandas`
- `jupyter`
- `scikit-learn`
- `mlxtend`
- `matplotlib`
- `seaborn`
- `graphviz`

### Using `conda`

A `conda` environment with these dependencies installed can be created via:

```bash
conda env create --file environment.yml
source activate madpy-ml-sklearn-2018
```

### Using `pip`

Inside a virtual environment, `pip` install the needed packages via:

```bash
pip install -r requirements.txt
```

Note: to render decision tree graphs [Graphviz](https://www.graphviz.org/download/) must also be installed.


## Usage

The slides for this talk can be generated via:

```bash
jupyter nbconvert machine-learning-sklearn-madpy.ipynb --to slides --post serve
```

To run the corresponding notebook use

```bash
jupyter notebook machine-learning-sklearn-madpy.ipynb
```
