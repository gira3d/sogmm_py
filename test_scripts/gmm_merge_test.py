from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.cluster import kmeans_plusplus
import matplotlib.pyplot as plt
import numpy as np
from termcolor import cprint

from tests_utils import *
from gmm_py import GMMf2CPU as GMMf2


def merge_gmm(g1, g2, n_samples1, n_samples2):
    new_means = np.concatenate((g1.means_, g2.means_))
    new_covs = np.concatenate(
        (g1.covariances_, g2.covariances_))
    new_precs = np.concatenate((g1.precisions_, g2.precisions_))
    new_precs_chol = np.concatenate(
        (g1.precisions_cholesky_, g2.precisions_cholesky_))

    new_weights = np.concatenate(
        (n_samples1 * g1.weights_, n_samples2 * g2.weights_))
    new_weights /= (n_samples1 + n_samples2)
    new_weights /= np.sum(new_weights)
    new_components = g1.n_components + g2.n_components

    g_new = GaussianMixture(
        n_components=new_components, init_params='k-means++', verbose=True)

    g_new.means_ = new_means
    g_new.covariances_ = new_covs
    g_new.precisions_ = new_precs
    g_new.precisions_cholesky_ = new_precs_chol
    g_new.weights_ = new_weights

    return g_new


cprint('loading data', 'grey')
n_components = 4
n_samples1 = 1800
n_samples2 = 200

D = []
X, _ = datasets.make_blobs(
    n_samples=n_samples1, centers=n_components, random_state=10)
D.append(X)

Y, _ = datasets.make_blobs(
    n_samples=n_samples2, centers=n_components, random_state=8)
D.append(Y)

cprint('running the python merging case', 'grey')
g = []
g.append(GaussianMixture(n_components=n_components,
                         init_params='k-means++', random_state=0))
g[0].fit(X)
g.append(GaussianMixture(n_components=n_components,
                         init_params='k-means++', random_state=0))
g[1].fit(Y)


merged_fig, merged_ax = plt.subplots()
g_new = merge_gmm(g[0], g[1], n_samples1, n_samples2)

colors = ['green', 'cyan']
for k in range(len(g)):
    merged_ax.scatter(D[k][:, 0], D[k][:, 1], c='blue', alpha=0.5,
                      edgecolors='black', label='orig')
    make_ellipses_gmm(g[k], merged_ax, case='py', color=colors[k])

    merged_ax.scatter(g[k].means_[:, 0], g[k].means_[:, 1],
                      s=100, c=colors[k], marker='o', linewidth=3)


sampled_fig, sampled_ax = plt.subplots()
sampled_data, _ = g_new.sample(n_samples1 + n_samples2)
sampled_ax.scatter(sampled_data[:, 0], sampled_data[:, 1], c='chocolate', alpha=0.5,
                   edgecolors='black')
make_ellipses_gmm(g_new, sampled_ax, case='py', color='red')
sampled_ax.scatter(g_new.means_[:, 0], g_new.means_[
                   :, 1], s=100, c='red', marker='o', linewidth=3)

cprint('running the cpp merging case', 'grey')
cg = []
cg.append(GMMf2(n_components, 1e-3, 1e-6, 100))
resp_ref1 = np.zeros((n_samples1, n_components))
_, indices = kmeans_plusplus(
    X,
    n_components,
    random_state=0,
)
resp_ref1[indices, np.arange(n_components)] = 1
cg[0].fit(X, resp_ref1)

cg.append(GMMf2(n_components, 1e-3, 1e-6, 100))
resp_ref2 = np.zeros((n_samples2, n_components))
_, indices = kmeans_plusplus(
    Y,
    n_components,
    random_state=0,
)
resp_ref2[indices, np.arange(n_components)] = 1
cg[1].fit(Y, resp_ref2)

merged_cpp_fig, merged_cpp_ax = plt.subplots()
colors = ['green', 'cyan']
for k in range(len(g)):
    merged_cpp_ax.scatter(D[k][:, 0], D[k][:, 1], c='blue', alpha=0.5,
                          edgecolors='black', label='orig')
    make_ellipses_gmm(cg[k], merged_cpp_ax, case='cpp', color=colors[k])

    merged_cpp_ax.scatter(cg[k].means_[:, 0], cg[k].means_[:, 1],
                          s=100, c=colors[k], marker='o', linewidth=3)

cg[0].merge(cg[1])

sampled_cpp_fig, sampled_cpp_ax = plt.subplots()
sampled_cpp_data = cg[0].sample(n_samples1 + n_samples2, 3.0)
sampled_cpp_ax.scatter(sampled_cpp_data[:, 0], sampled_cpp_data[:, 1], c='chocolate', alpha=0.5,
                       edgecolors='black')
make_ellipses_gmm(cg[0], sampled_cpp_ax, case='cpp', color='red')
sampled_cpp_ax.scatter(cg[0].means_[:, 0], cg[0].means_[:, 1],
                       s=100, c='red', marker='o', linewidth=3)
plt.show()
