from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from mnist import MNIST

# Import data from the folder
mndata = MNIST('./samples')

xtrain, ytrain = mndata.load_training()
xtest, ytest = mndata.load_testing()

# Convert one dimensional labels from arrays to lists
# in order to fit them in the model later
ytrain = ytrain.tolist()
ytest = ytest.tolist()

# # Check the first 10 images
# model = KNeighborsClassifier(n_neighbors=1)
# model.fit(xtrain, ytrain)

# pred = model.predict(xtest)
# print('The first 10 predicted values are:', pred[:10].tolist())
# print('The first 10 actual values are:', ytest[:10])

for k in [1, 3, 5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(xtrain, ytrain)

    pred = model.predict(xtest)

    score = accuracy_score(ytest, pred)

    print('For k =', k, 'we have accuracy:', 100*score, '%')
