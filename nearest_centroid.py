from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score

from mnist import MNIST

# Import data from the folder
mndata = MNIST('../samples')

xtrain, ytrain = mndata.load_training()
xtest, ytest = mndata.load_testing()

# Convert one dimensional labels from arrays to lists
ytrain = ytrain.tolist()
ytest = ytest.tolist()

model = NearestCentroid()
model.fit(xtrain, ytrain)

pred = model.predict(xtest)

acc = accuracy_score(ytest, pred)

print('For the nearest centroid algorithm the accuracy is:', acc*100, '%')
