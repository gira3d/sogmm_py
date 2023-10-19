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
from gmm_py import GMMf2CPU as GMMf2

cprint('loading data', 'grey')
n_samples = 12000
n_components = 4

X, _ = datasets.make_blobs(
    n_samples=n_samples, centers=n_components, random_state=10)

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
ax.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5,
           edgecolors='black', label='orig')
make_ellipses_gmm(g, ax, case='py')

for i, j in g.means_:
    ax.scatter(i, j, s=100, c='black', marker='o', linewidth=3)

# load initial resp from KMeans++ for the cpp gmm
resp_ref = np.zeros((n_samples, n_components))
kmeans_start = time()
_, indices = kmeans_plusplus(
    X,
    n_components,
    random_state=0,
)
kmeans_end = time()
cprint('kmeans_plusplus python time %f seconds' %
       (kmeans_end - kmeans_start), 'green')
resp_ref[indices, np.arange(n_components)] = 1

cprint('fitting GMM in cpp', 'grey')
cg = GMMf2(n_components, 1e-3, 1e-6, 100)
cpp_gmm_fit_start = time()
cg.fit(X, resp_ref)
cpp_gmm_fit_end = time()
cprint('time taken by cpp for the fit %f seconds' %
       (cpp_gmm_fit_end - cpp_gmm_fit_start), 'red')

# start test pickling
with open('./gmm_cpp.pkl', 'wb') as f:
    pickle.dump(cg, f)

with open('./gmm_cpp.pkl', 'rb') as f:
    cg_loaded = pickle.load(f)

assert(cg_loaded.n_components_ == cg.n_components_)
assert(cg_loaded.tol_ == cg.tol_)
assert(cg_loaded.reg_covar_ == cg.reg_covar_)
assert(cg_loaded.max_iter_ == cg.max_iter_)

np.testing.assert_array_almost_equal(cg_loaded.weights_, cg.weights_)
np.testing.assert_array_almost_equal(cg_loaded.means_, cg.means_)
np.testing.assert_array_almost_equal(cg_loaded.covariances_, cg.covariances_)
np.testing.assert_array_almost_equal(
    cg_loaded.precisions_cholesky_, cg.precisions_cholesky_)

os.remove('./gmm_cpp.pkl')
# end test pickling

make_ellipses_gmm(cg, ax, case='cpp')

for i, j in cg.means_:
    ax.scatter(i, j, s=100, c='red', marker='o', linewidth=3)

# start test sampling
cpp_samps = cg.sample(12000, 3.0)
samples_fig, samples_ax = plt.subplots()
samples_ax.scatter(cpp_samps[:, 0], cpp_samps[:, 1], c='grey', alpha=0.5,
                   edgecolors='black', label='orig')
samples_ax.set_aspect("equal")
# end test sampling

# likelihood score test start
X_test, _ = datasets.make_blobs(
    n_samples=n_samples, centers=n_components, random_state=2)
py_score = g.score(X_test)
cpp_score = cg.score(X_test)
np.testing.assert_array_almost_equal(py_score, cpp_score, decimal=4)
# likelihood score test end

plt.show()


# let the testing begin
cprint('testing C++ GMM functions', 'grey')

# construct another cpp gmm for individual function testing
cpp_gmm = GMMf2(n_components, 1e-3, 1e-6, 100)
cpp_gmm.weights_ = g.weights_
cpp_gmm.means_ = g.means_
# for cpp, since we are using Eigen which does not support tensors.
# need to flatten the tensors from numpy for testing
cpp_gmm.covariances_ = tensor_to_matrix(g.covariances_)
cpp_gmm.precisions_cholesky_ = tensor_to_matrix(g.precisions_cholesky_)

t1 = time()
python_output = _compute_log_det_cholesky(g.precisions_cholesky_, 'full', 2)
t2 = time()
cpp_output = cpp_gmm.compute_log_det_cholesky(cpp_gmm.precisions_cholesky_)
t3 = time()
np.testing.assert_array_almost_equal(python_output, cpp_output, decimal=4)
cprint('compute_log_det_cholesky test passed. python %f seconds, cpp %f seconds' %
       (t2 - t1, t3 - t2), 'green')

t1 = time()
python_output = _compute_precision_cholesky(g.covariances_, 'full')
t2 = time()
cpp_output = cpp_gmm.compute_precision_cholesky(cpp_gmm.covariances_)
t3 = time()
cpp_output_tensored = matrix_to_tensor(cpp_output)
np.testing.assert_array_almost_equal(
    python_output, cpp_output_tensored, decimal=4)
cprint('compute_precision_cholesky test passed. python %f seconds, cpp %f seconds' %
       (t2 - t1, t3 - t2), 'green')

t1 = time()
python_output = _estimate_log_gaussian_prob(
    X, g.means_, g.precisions_cholesky_, 'full')
t2 = time()
cpp_output = cpp_gmm.estimate_log_gaussian_prob(
    X, g.means_, cpp_gmm.precisions_cholesky_)
t3 = time()
np.testing.assert_array_almost_equal(python_output, cpp_output, decimal=4)
cprint('estimate_log_gaussian_prob test passed up to 4 decimal places. python %f seconds, cpp %f seconds' %
       (t2 - t1, t3 - t2), 'green')

t1 = time()
python_output = g._estimate_log_prob(X)
t2 = time()
cpp_output = cpp_gmm.estimate_log_prob(X)
t3 = time()
np.testing.assert_array_almost_equal(python_output, cpp_output, decimal=4)
cprint('estimate_log_prob test passed up to 4 decimal places. python %f seconds, cpp %f seconds' %
       (t2 - t1, t3 - t2), 'green')

t1 = time()
python_output = g._estimate_weighted_log_prob(X)
t2 = time()
cpp_output = cpp_gmm.estimate_weighted_log_prob(X)
t3 = time()
np.testing.assert_array_almost_equal(python_output, cpp_output, decimal=4)
cprint('estimate_weighted_log_prob test passed up to 4 decimal places. python %f seconds, cpp %f seconds' %
       (t2 - t1, t3 - t2), 'green')

t1 = time()
python_output1, python_output2 = g._estimate_log_prob_resp(X)
t2 = time()
cpp_output1, cpp_output2 = cpp_gmm.estimate_log_prob_resp(X)
t3 = time()
np.testing.assert_array_almost_equal(
    python_output1.reshape((n_samples, 1)), cpp_output1, decimal=4)
np.testing.assert_array_almost_equal(python_output2, cpp_output2, decimal=4)
cprint('estimate_log_prob_resp test passed up to 4 decimal places. python %f seconds, cpp %f seconds' %
       (t2 - t1, t3 - t2), 'green')

t1 = time()
python_output1, log_resp = g._e_step(X)
t2 = time()
cpp_output1, cpp_output2 = cpp_gmm.e_step(X)
t3 = time()
np.testing.assert_array_almost_equal(python_output1, cpp_output1, decimal=4)
np.testing.assert_array_almost_equal(log_resp, cpp_output2, decimal=4)
cprint('e_step test passed up to 4 decimal places. python %f seconds, cpp %f seconds' %
       (t2 - t1, t3 - t2), 'green')

t1 = time()
python_output1, python_output2, python_output3 = _estimate_gaussian_parameters(
    X, np.exp(log_resp), 1e-6, 'full')
t2 = time()
cpp_output1, cpp_output2, cpp_output3 = cpp_gmm.estimate_gaussian_parameters(
    X, np.exp(log_resp))
t3 = time()
np.testing.assert_array_almost_equal(python_output1, cpp_output1, decimal=3)
np.testing.assert_array_almost_equal(python_output2, cpp_output2, decimal=3)
np.testing.assert_array_almost_equal(
    python_output3, matrix_to_tensor(cpp_output3), decimal=3)
cprint('estimate_gaussian_parameters test passed up to 3 decimal places. python %f seconds, cpp %f seconds' %
       (t2 - t1, t3 - t2), 'green')

t1 = time()
# use the log_resp output from the e_step test
g._m_step(X, np.exp(log_resp))
python_output1 = g.weights_
python_output2 = g.means_
python_output3 = g.covariances_
python_output4 = g.precisions_cholesky_
t2 = time()
cpp_gmm.m_step(X, np.exp(log_resp))
cpp_output1 = cpp_gmm.weights_
cpp_output2 = cpp_gmm.means_
cpp_output3 = cpp_gmm.covariances_
cpp_output4 = cpp_gmm.precisions_cholesky_
t3 = time()
np.testing.assert_array_almost_equal(python_output1, cpp_output1, decimal=2)
np.testing.assert_array_almost_equal(python_output2, cpp_output2, decimal=2)
np.testing.assert_array_almost_equal(
    python_output3, matrix_to_tensor(cpp_output3), decimal=2)
np.testing.assert_array_almost_equal(
    python_output4, matrix_to_tensor(cpp_output4), decimal=2)
cprint('m_step test passed up to 3 decimal places. python %f seconds, cpp %f seconds' %
       (t2 - t1, t3 - t2), 'green')
