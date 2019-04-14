import struct
import numpy as np
from sklearn import neighbors, metrics
import matplotlib.pyplot as plt
import pandas as pd
import itertools

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

data = pd.read_csv("train.csv").values
data_test = pd.read_csv("test.csv").values
data_train = data[:,1:]
train_label = data[:,0]
# test_label = data_test[:,0]

# idx = (train_label == 2) | (train_label == 3) | (train_label == 8)
# X = data_train[idx]
# Y = train_label[idx]
knn = neighbors.KNeighborsClassifier(n_neighbors = 3).fit(data_train,train_label)

# idx = (test_label == 2) | (test_label == 3) | (test_label == 8)
# x_test = data_test[idx]
# y_true = test_label[idx]
y_pred = knn.predict(data_test)

# def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting 'normalize = True'.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print('Normalized confusion matrix')
#     else:
#         print('Confusion matix, without normalization')
    
#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                     horizontalalignment = 'center',
#                     color = 'white' if cm[i, j] > thresh else 'black')

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')


# cm = metrics.confusion_matrix(y_true, y_pred)
# plot_confusion_matrix(cm, ['2','3','8'],normalize=True)

# show figure that predict as 2 but actual is 8
# idx = np.where((y_pred == 2) & (y_true == 8))[0]
# fig = plt.figure(figsize=(5,30))
# for i in range(len(idx)):
#     # ax = fig.add_subplot(len(idx), 1, i+1)
#     # imgplot = ax.imshow(np.reshape(x_test[idx[i],:], (28, 28)), cmap = plt.cm.get_cmap('Greys'))
#     # imgplot.set_interpolation('nearest')
#     d = x_test[idx[i],1:]
#     d.shape = (28,28)
#     plt.imshow(255 - d, cmap = 'gray')
#     plt.show()


# d = xtest[8]
# d.shape = (28,28)
# pt.imshow(255 - d, cmap = 'gray')
# print(clf.predict([xtest[8]]))
# pt.show()

# determine accuracy
# count = 0
# for i in range(0,len(x_test)):
#     count += 1 if y_pred[i] == y_true[i] else 0
# print('Accuracy=', (count/len(x_test))*100)

# add id to predict array
index = np.array(range(1,len(data_test)+1))
p_withindex = np.c_[index, y_pred]

# add title to predict array
title = ['ImageId', 'Label']
p_withtitle = np.vstack([title,p_withindex])

np.savetxt('submission(knn).csv', [p for p in p_withtitle], delimiter=',', fmt='%s')
