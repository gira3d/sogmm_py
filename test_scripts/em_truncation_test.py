import os
import pickle
from sklearn import datasets
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.mixture._gaussian_mixture import _compute_log_det_cholesky
from sklearn.cluster import kmeans_plusplus
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture as PythonGMM
from termcolor import cprint
from time import time

from tests_utils import *
from gmm_py import GMMf2

cprint('loading data', 'grey')
n_samples = 4000
n_components = 500

X, _ = datasets.make_blobs(
    n_samples=n_samples, centers=n_components, random_state=10)

np.savetxt('../test_data/FitSmallDim2/data.csv', X, delimiter=',')

# python gmm
cprint('fitting GMM in python', 'grey')
g = PythonGMM(n_components=n_components,
              init_params='k-means++', random_state=0)
python_gmm_fit_start = time()
g.fit(X)
python_gmm_fit_end = time()
cprint('time taken by python for the fit %f seconds' %
       (python_gmm_fit_end - python_gmm_fit_start), 'red')
fig, ax = plt.subplots()
#ax.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5,
#           edgecolors='black', label='orig')
#make_ellipses_gmm(g, ax, case='py')

#for i, j in g.means_:
#    ax.scatter(i, j, s=100, c='black', marker='o', linewidth=3)

# load initial resp from KMeans++ for the cpp gmm
kmeans_start = time()
_, indices = kmeans_plusplus(
    X,
    n_components,
    random_state=0,
)
kmeans_end = time()
cprint('kmeans_plusplus python time %f seconds' %
       (kmeans_end - kmeans_start), 'green')

max_iterations = 101
cpp_scores = np.zeros((max_iterations-1,1))
for n_iterations in range(1, max_iterations):
    resp_ref = np.zeros((n_samples, n_components))
    resp_ref[indices, np.arange(n_components)] = 1

    cprint('fitting GMM in cpp with ' + str(n_iterations) + ' iterations', 'grey')
    cg = GMMf2(n_components, 1e-3, 1e-6, n_iterations)
    cpp_gmm_fit_start = time()
    cg.fit(X, resp_ref)
    cpp_gmm_fit_end = time()
    cprint('time taken by cpp for the fit %f seconds' %
           (cpp_gmm_fit_end - cpp_gmm_fit_start), 'red')

    #make_ellipses_gmm(cg, ax, case='cpp')

    #for i, j in cg.means_:
    #    ax.scatter(i, j, s=100, c='red', marker='o', linewidth=3)

    # likelihood score test start
    #X_test, _ = datasets.make_blobs(
    #    n_samples=n_samples, centers=n_components, random_state=2)
    py_score = g.score(X)
    cpp_score = cg.score(X)
    print(py_score)
    print(cpp_score)
    cpp_scores[n_iterations-1,:] = cpp_score
    # likelihood score test end

print(cpp_scores)
plt.plot(np.arange(0,max_iterations-1), cpp_scores)
plt.xlabel("Iteration")
plt.ylabel("Score")
plt.show()
