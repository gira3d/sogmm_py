import numpy as np
from termcolor import cprint
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import kmeans_plusplus
import time

from kinit_py import KInitf2
from gmm_py import GMMf2
from tests_utils import make_ellipses_gmm

n_samples = 9000
n_components = 7
X, _ = datasets.make_blobs(
    n_samples=n_samples, centers=n_components, random_state=10)
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5,
           edgecolors='black', label='orig')

resp_ref = np.zeros((n_samples, n_components))
start = time.time()
centers, indices = kmeans_plusplus(
    X,
    n_components,
    random_state=10,
)
end = time.time()
cprint('python time taken %f seconds' % (end - start), 'red')
resp_ref[indices, np.arange(n_components)] = 1
gmm1 = GMMf2(n_components)
success = gmm1.fit(X, resp_ref)

ax.scatter(centers[:, 0], centers[:, 1], c='red', alpha=1.0, edgecolors='black', s=100)
make_ellipses_gmm(gmm1, ax, case='cpp', color='red')

resp_cpp = np.zeros((n_samples, n_components))
kinit = KInitf2(n_components)
start = time.time()
centers_cpp, indices_cpp = kinit.resp_calc(X)
end = time.time()
cprint('cpp time taken %f seconds' % (end - start), 'green')
resp_cpp[indices_cpp, np.arange(n_components)] = 1
gmm2 = GMMf2(n_components)
success = gmm2.fit(X, resp_cpp)

ax.scatter(centers_cpp[:, 0], centers_cpp[:, 1], c='green', alpha=1.0, edgecolors='black', s=100)
make_ellipses_gmm(gmm2, ax, case='cpp', color='green')


plt.show()