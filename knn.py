from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from mnist import MNIST

# Import data from the folder
mndata = MNIST('../samples')

xtrain, ytrain = mndata.load_training()
xtest, ytest = mndata.load_testing()

# Convert one dimensional labels from arrays to lists
ytrain = ytrain.tolist()
ytest = ytest.tolist()

# Find the k = 1 nearest neighbour
acc = [None] * 2
for k in [1, 3]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(xtrain, ytrain)

    pred = model.predict(xtest)

    score = accuracy_score(ytest, pred)

    acc.append(score)

    print('For k =', k, 'we have accuracy:', 100*score, '%')
