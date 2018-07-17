import numpy as np
from PIL import Image
from numpy import arange
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier


def getBounds(nparray):
    rows = nparray.shape[0]
    cols = nparray.shape[1]

    xmin = rows
    ymin = cols
    xmax = 0
    ymax = 0

    for x in range(rows):
        for y in range(cols):
            if (nparray[x, y] != 0):
                if x < xmin:
                    xmin = x
                elif x > xmax:
                    xmax = x
                if y < ymin:
                    ymin = y
                elif y > ymax:
                    ymax = y
# Special thanks to https://stackoverflow.com/questions/32682754/np-delete-and-np-s-whats-so-special-about-np-s
    nparray = np.delete(nparray, np.s_[0: xmin], axis=0)
    nparray = np.delete(nparray, np.s_[0: ymin], axis=1)
    nparray = np.delete(nparray, np.s_[xmax-xmin: -1], axis=0)
    nparray = np.delete(nparray, np.s_[ymax-ymin: -1], axis=1)
    
    return nparray


def reshapeImg(nparray):
    newarray = getBounds(nparray.reshape(28, 28))
    img = Image.fromarray(newarray)
    img = img.resize((20, 20))
    finalarray = np.array(img)
    return finalarray


# Special thanks to https://gist.github.com/fxsjy/5574345 for understanding steps in the process
mnist = datasets.fetch_mldata("MNIST Original") # , data_home='C:/Users/Daniel/PycharmProjects/498AML/HW1/') # In order for this to work, I had to source the data_home directly. It may just be an issue on my end though
                                                # I also processed this using the mnist-data.mat file

train_indexes = 60000
test_indexes = 10000

train_dimensions = arange(0, train_indexes)
test_dimensions = arange(train_indexes + 1, train_indexes+test_indexes)

X_train, y_train = mnist.data[train_dimensions], mnist.target[train_dimensions]
X_test, y_test = mnist.data[test_dimensions], mnist.target[test_dimensions]

X_train[X_train < 128] = 0
X_test[X_test < 128] = 0

print "Stretching X_train\n"
stretched_train = []
for nparray in X_train:
    newarr = reshapeImg(nparray)
    stretched_train.append(np.ravel(newarr))

print "Stretching X_test\n"
stretched_test = []
for nparray in X_test:
    newarr = reshapeImg(nparray)
    stretched_test.append(np.ravel(newarr))

print "Untouched Gaussian Accuracy"
gb = GaussianNB().fit(X_train, y_train)
gpred = gb.predict(X_test)
print accuracy_score(y_test, gpred, normalize=True), "\n"

print "Untouched Bernoulli Accuracy: "
bb = BernoulliNB(alpha=1, binarize=0).fit(X_train, y_train)
bpred = bb.predict(X_test)
print accuracy_score(y_test, bpred, normalize=True), "\n"

print "Stretched Gaussian Accuracy"
sgb = GaussianNB().fit(stretched_train, y_train)
spred = sgb.predict(stretched_test)
print accuracy_score(y_test, spred, normalize=True), "\n"

print "Stretched Bernoulli Accuracy"
sbb = BernoulliNB(alpha=1, binarize=0).fit(stretched_train, y_train)
sbpred = sbb.predict(stretched_test)
print accuracy_score(y_test, sbpred, normalize=True), "\n"

# ---------------------------------UNTOUCHED-------------------------------------- #
def untouchedRandomTree(trees, depth):
    usdf = RandomForestClassifier(n_estimators=trees, max_depth=depth).fit(X_train, y_train)
    usdf_pred = usdf.predict(X_test)
    print "Accuracy of Untouched Random Trees =", trees, ", Depth =", depth
    print accuracy_score(y_test, usdf_pred, normalize=True), "\n"


untouchedRandomTree(10, 4)
untouchedRandomTree(20, 4)
untouchedRandomTree(30, 4)
untouchedRandomTree(10, 8)
untouchedRandomTree(20, 8)
untouchedRandomTree(30, 8)
untouchedRandomTree(10, 16)
untouchedRandomTree(20, 16)
untouchedRandomTree(30, 16)
# ------------------------------------STRETCHED----------------------------------------- #
def stretchedRandomTree(trees, depth):
    sdf = RandomForestClassifier(n_estimators=trees, max_depth=depth).fit(stretched_train, y_train)
    sdf_pred = sdf.predict(stretched_test)
    print "Accuracy of Stretched Random Trees =", trees, ", Depth =", depth
    print accuracy_score(y_test, sdf_pred, normalize=True), "\n"


stretchedRandomTree(10, 4)
stretchedRandomTree(20, 4)
stretchedRandomTree(30, 4)
stretchedRandomTree(10, 8)
stretchedRandomTree(20, 8)
stretchedRandomTree(30, 8)
stretchedRandomTree(10, 16)
stretchedRandomTree(20, 16)
stretchedRandomTree(30, 16)

print "end"

