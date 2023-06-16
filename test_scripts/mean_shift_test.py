import numpy as np
from sklearn import datasets
from sklearn.cluster import MeanShift as PythonMS
from sklearn.neighbors import NearestNeighbors
from termcolor import cprint
from time import time
import matplotlib.pyplot as plt

from mean_shift_py import MeanShift as CppMS

n_samples = 12000
n_components = 4

X, _ = datasets.make_blobs(
    n_samples=n_samples, centers=n_components, random_state=10)

b = 3.5

clustering_fig, clustering_ax = plt.subplots(2, 1)

start = time()
skl_ms = PythonMS(bandwidth=b, bin_seeding=True)
skl_ms.fit(X)
skl_lbls = skl_ms.labels_
skl_clstr_cntrs = skl_ms.cluster_centers_
skl_num_clstrs = len(np.unique(skl_lbls))
end = time()
cprint('Time taken for Python mean shift fit %f seconds' %
       (end - start), 'red')

cprint('PythonMS number of clusters: %d' %
       skl_num_clstrs, 'green')

for k in range(skl_num_clstrs):
    my_members = skl_lbls == k
    cluster_center = skl_clstr_cntrs[k]
    clustering_ax[0].scatter(X[my_members, 0], X[my_members, 1],
                             s=40, color=np.random.rand(3,), edgecolors='black')

clustering_ax[0].set_title('python sklearn meanshift output')
start = time()
cpp_ms = CppMS(b)
cpp_ms.fit(X)
end = time()
num_clusters = cpp_ms.get_num_modes()
cprint('Time taken for C++ mean shift fit %f seconds' % (end - start), 'red')
cprint('CppMS number of clusters: %d' %
       num_clusters, 'green')
cluster_centers = cpp_ms.get_mode_centers()
nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(cluster_centers)
labels = np.zeros(X.shape[0], dtype=int)
distances, idxs = nbrs.kneighbors(X)
labels = idxs.flatten()

for k in range(num_clusters):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    clustering_ax[1].scatter(X[my_members, 0], X[my_members, 1],
                             s=40, color=np.random.rand(3,), edgecolors='black')

clustering_ax[1].set_title('c++ meanshift output')

clustering_ax[0].set_aspect("equal")
clustering_ax[1].set_aspect("equal")
plt.tight_layout()
plt.show()
