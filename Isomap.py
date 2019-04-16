import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('train.csv').values
train_data = data[:,1:]
train_label = data[:,0]

X = train_data[train_label == 8]
Y = manifold.Isomap(n_neighbors = 5, n_components = 2).fit_transform(X)


plt.scatter(Y[:,0], Y[:,1])

def find_landmarks(Y, n, m):
    xr = np.linspace(np.min(Y[:,0]), np.max(Y[:,0]),n)
    yr = np.linspace(np.min(Y[:,1]), np.max(Y[:,1]),n)
    xg, yg = np.meshgrid(xr, yr)

    idx = [0]*(n*m)

    for i, x, y in zip(range(n*m), xg.flatten(), yg.flatten()):
        idx[i] = int(np.sum(np.abs(Y - np.array([x,y]))**2, axis = -1 ).argmin())
    
    return idx

landmarks = find_landmarks(Y, 5, 5)

plt.scatter(Y[:,0], Y[:,1])
plt.scatter(Y[landmarks,0], Y[landmarks,1])

fig = plt.figure(figsize=(15,15))
for i in range(len(landmarks)):
    ax = fig.add_subplot(5, 5, i+1)
    imgplot = ax.imshow(np.reshape(X[landmarks[i]], (28,28)), cmap=plt.cm.get_cmap("Greys"))
    imgplot.set_interpolation("nearest")

plt.show()