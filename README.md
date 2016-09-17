# ML-Python

This project contains my attempts to implement machine learning algorithms with Python. These implementations are not supposed to be optimized.

## Prerequisites

Project was written with Python 3.5.2

## Installation

There are some dependencies such as Numpy and matplotlib. They can be installed via pip:
```
pip install -r requirements.txt
```

## Running examples

This project contains different solutions, every one is located in separate folder. There are run.py files which can be executed to see algorithms in action:

```
python run.py
```

# Problem set

## Linear regression

To test implementation [Concrete Compressive Strength Data Set](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength) is used. Data set is devided on training and test subsets. 2D projection is used to visually show results.
Also 1D synthetic data is generated to demonstrate more crearly how predictions will be looks like.

**Results:**
Normal equation and gradient descent works well on small data sets.
Normal equation is quite accurate but there are some problems with performance on very large data sets with many features.
Gradient descent provides approximate solution but it can be more effective on large data sets.

## Logistical regression

There is some randomly generated synthetic data to test algorithm with linear and polynominal models.
Real data set: [Haberman's Survival Data Set](https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival)

**Results:**
With some twiking of params this implementation can have 75-80% accuracy on real dataset. main problem is that implementation of optimization algorithm (gradient descent) if falling in local minimum so another optimiation is required for better results.
Regularization tests shows
