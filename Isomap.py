import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('train.csv').values
train_data = data[:,1:]
train_label = data[:,0]

X = train_data[train_label == 8]
Y = manifold.Isomap(n_neighbors = 5, n_components = 2).fit_transform(X)
