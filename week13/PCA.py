import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import torchvision.datasets as datasets

def plot_images(dat, k=20, size=28):
    """ Plot the first k vectors as 28 x 28 images """
    x2 = dat[0:k,:].reshape(-1, size, size)
    x2 = x2.transpose(1, 0, 2)
    fig, ax = plt.subplots(figsize=(20,12))
    ax.imshow(x2.reshape(size, -1), cmap='bone')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.show()

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
# Load full dataset
X = mnist_trainset.train_data.numpy().reshape((60000, 28*28))
y = mnist_trainset.train_labels.numpy()

# Take out random subset
rp = np.random.permutation(len(y))
X = X[rp[:5000], :]
y = y[rp[:5000]]

### YOUR CODE HERE
n_components = 784
pca = PCA(n_components)
pca.fit(X)
#plot_images(pca.components_[:20])

#plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_))
#plt.show()
c = pca.get_covariance()
print('done')

### END CODE