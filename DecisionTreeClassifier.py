import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random

data_train = pd.read_csv("train.csv").values
data_test = pd.read_csv('test.csv').values
data_test_length = len(data_test)
clf = DecisionTreeClassifier()

# training dataset
xtrain = data_train[:, 1:]
train_label = data_train[:,0]
clf.fit(xtrain, train_label)

# testing dataset
xtest = data_test

# testing with xtest[8]
# d = xtest[8]
# d.shape = (28,28)
# pt.imshow(255 - d, cmap = 'gray')
# print(clf.predict([xtest[8]]))
# pt.show()

# determine accuracy
# count = 0
# for i in range(0,data_test_length):
#     count += 1 if p[i] == actual_label[i] else 0
# print('Accuracy=', (count/data_test_length)*100)

# predict test data label
p = clf.predict(xtest)

# print test data label
# print('ImageId, Label')
# for i in range(0,len(xtest)):
#     print(i+1, p[i])

# add id to predict array
index = np.array(range(1,len(xtest)+1))
p_withindex = np.c_[index, p]

# add title to predict array
title = ['ImageId', 'Label']
p_withtitle = np.vstack([title,p_withindex])

np.savetxt('submission.csv', [p for p in p_withtitle], delimiter=',', fmt='%s')
