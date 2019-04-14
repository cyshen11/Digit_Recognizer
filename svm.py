import struct
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import itertools

data = pd.read_csv("train.csv").values
X_sub = pd.read_csv("test.csv").values
X = data[:,1:]
y = data[:,0]
X = X/255.0

# Set the parameters by cross-validation
# Not enough memory to tune parameter
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='f1_macro')
# clf.fit(X, Y)

# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)

svc = SVC(C=5, gamma=0.05).fit(X,y)
y_pred = svc.predict(X_sub)

# Print f1 score
print(metrics.f1_score(y, y_pred, average='weighted'))

# add id to predict array
# index = np.array(range(1,len(X)+1))
# p_withindex = np.c_[index, y_pred]

# # add title to predict array
# title = ['ImageId', 'Label']
# p_withtitle = np.vstack([title,p_withindex])

# np.savetxt('submission(svm).csv', [p for p in p_withtitle], delimiter=',', fmt='%s')
