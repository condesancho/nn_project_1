from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score

from mnist import MNIST

import time

# Import data from the folder
mndata = MNIST('./samples')

xtrain, ytrain = mndata.load_training()
xtest, ytest = mndata.load_testing()

# Convert one dimensional labels from arrays to lists
ytrain = ytrain.tolist()
ytest = ytest.tolist()

# Start timer
start_time = time.time()

model = NearestCentroid()
model.fit(xtrain, ytrain)

pred = model.predict(xtest)

# # Check the first 10 images
print('The first 10 predicted values are:', pred[:10].tolist())
print('The first 10 actual values are:', ytest[:10])

acc = accuracy_score(ytest, pred)

print('For the nearest centroid algorithm the accuracy is:', acc*100, '%')
print("Time passed: %s seconds." % (time.time() - start_time))
