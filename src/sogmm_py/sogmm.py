"""Self-Organizing Gaussian Mixture Models."""

import copy
import numpy as np
import pickle
from scipy.stats import multivariate_normal

from sogmm_py.utils import check_random_state, matrix_to_tensor, np_to_o3d, tensor_to_matrix

from mean_shift_py import MeanShift as MSf2
from kinit_py import KInitf4CPU
from gmm_py import GMMf4CPU
try:
    from gmm_open3d_py import GMMf4GPU
except ImportError:
    print('Could not import GPU codebase.')
    print('Either it is macOS or you did not compile the GPU support for SOGMM.')


class SOGMM(object):
    """Self-Organizing Gaussian Mixture Model.

    Attributes
    ----------
    bandwidth : float
        Bandwidth parameter for PRI/MeanShift -- the first stage of SOGMM system.
    random_state : int or RandomState
        Controls the randomness in K-Means++ stage of the SOGMM system and sampling.
    model : gmm_py.GMMf4
        Overall SOGMM model created using the point clouds received so far.
    """

    def __init__(self, bandwidth=0.02, random_state=None, compute='CPU'):
        """
        Parameters
        ----------
        bandwidth : float, optional (default: 0.02)
            Bandwidth parameter for PRI/MeanShift.
        random_state : int or RandomState, optional (default: global random state)
            If an integer is given, it fixes the seed. Defaults to the global numpy
            random number generator.
        """

        self.bandwidth = bandwidth
        self.random_state = check_random_state(random_state)
        self.model = None
        self.compute = compute
        self.latest_model = None

    def fit(self, pcld):
        """Fit SOGMM over a 4-D point cloud.

        PRI/MeanShift is used to decide number of components for the input point cloud.
        Using this number of components, K-Means++ is used to initialize
        Expectation-Maximization (EM) over the point cloud.

        Parameters
        ----------
        pcld : array-like, shape (n_samples, 4)
            Input point cloud with 4 features (usually, the 3D point coordinates in world frame
            and the grayscale color associated with them).

        Returns
        -------
        local_model : gmm_py.GMMf4
            Local SOGMM model for this pcld
        """

        # PRI/MeanShift
        d = np.array([np.linalg.norm(x) for x in pcld[:, 0:3]])[:, np.newaxis]
        g = pcld[:, 3][:, np.newaxis]
        ms_data = np.concatenate((d, g), axis=1)

        ms = MSf2(self.bandwidth)
        ms.fit(ms_data)
        n_components = ms.get_num_modes()

        return self.gmm_fit(pcld, n_components)

    def gmm_fit(self, pcld, n_components):
        """Fit a GMM model with the passed number of components and update the model.

        Parameters
        ----------
        pcld : array-like, shape (n_samples, 4)
            Input point cloud with 4 features (usually, the 3D point coordinates in world frame
            and the grayscale color associated with them).
        n_components : int
            Number of components to use for the fit
        """

        n_samples = pcld.shape[0]

        # K-Means++
        resp = np.zeros((n_samples, n_components))
        kinit = KInitf4CPU()
        _, indices = kinit.resp_calc(
            pcld, n_components)
        resp[indices, np.arange(n_components)] = 1

        # Expectation-Maximization (EM) to get the local model
        local_model = None
        success = False
        if self.compute == 'CPU':
            print('Compute platform is CPU')
            local_model = GMMf4CPU(n_components)
            success = local_model.fit(pcld, resp)
        elif self.compute == 'GPU':
            print('Compute platform is GPU')
            local_model = GMMf4GPU(n_components, n_samples)
            success = local_model.fit(pcld, resp)
        else:
            print('No compute platform specified, running on the CPU')
            local_model = GMMf4CPU(n_components)
            success = local_model.fit(pcld, resp)

        if success:
            self.latest_model = local_model

        if self.model is None and success:
            self.model = copy.deepcopy(local_model)
            return self.model
        elif self.model is not None and success:
            self.model.merge(local_model)
            return self.model
        else:
            print('The EM run during SOGMM modeling did not succeed. Exiting.')
            return None

    def color_conditional(self, x):
        """Expected color using conditional distribution \\( p(c | X = x) \\) over the full
        model so far. Also return the associate variance estimate.

        Parameters
        ----------
        x : array-like, shape (n_samples, 3)
            Points in the space that we know.

        Returns
        -------
        expected_value : array-like, shape (n_samples, 1)
            Expected value of the color at each of the points
        variance : array-like, shape (n_samples, 1)
            Variance (degree of confidence) about the color at each of the points
        """
        n_samples = x.shape[0]
        n_components = self.model.n_components_
        covariances_tensor = matrix_to_tensor(self.model.covariances_, 4)

        weights = np.zeros((n_samples, n_components))
        means = np.zeros((n_samples, n_components))
        vars = np.zeros((n_components))
        for k in range(n_components):
            mu_kX = self.model.means_[k, 0:3]
            mu_kli = self.model.means_[k, 3]
            sigma_kXX = covariances_tensor[k, 0:3, 0:3]
            sigma_kXli = covariances_tensor[k, 0:3, 3]
            sigma_kliX = covariances_tensor[k, 3, 0:3]
            sigma_klili = covariances_tensor[k, 3, 3]

            weights[:, k] = self.model.weights_[k] * \
                multivariate_normal.pdf(x, mu_kX, sigma_kXX)
            means[:, k] = mu_kli + np.dot(sigma_kliX, np.dot(
                np.linalg.inv(sigma_kXX), (x - mu_kX).T))
            vars[k] = sigma_klili - \
                np.dot(sigma_kliX, np.dot(np.linalg.inv(sigma_kXX), sigma_kXli))

        weights[weights < self.model.reg_covar_] = 0.0

        weights_sums = weights.sum(axis=1)
        normalized_weights = weights / weights_sums[:, np.newaxis]

        np.nan_to_num(normalized_weights, nan=0.0, copy=False)

        expected_values = np.zeros((n_samples, 1))
        expected_values = np.multiply(normalized_weights,
                                      means).sum(axis=1, keepdims=True)

        uncerts = np.zeros((n_samples, 1))
        uncerts = np.multiply(normalized_weights, np.square(means) +
                              vars).sum(axis=1, keepdims=True)
        uncerts = uncerts - np.square(expected_values)

        print('uncerts contains nans?', np.isnan(uncerts).any())

        return expected_values

    def joint_dist_sample(self, num_samples, sigma=2.5):
        """Sample from the joint distribution.

        Randomly samples n_samples points out of the joint distribution
        \\( p (x, y, z, c) \\) within a sigma-confidence region.

        Parameters
        ----------
        num_samples : int
            Desired number of samples.
        sigma : float, optional (default: 2.5)
            Confidence region to limit random sampling.

        Returns
        -------
        sampled_points : array-like, shape (num_samples, 4)
            Points sampled our of the joint distribution \\( p (x, y, z, c) \\).
        """

        return self.model.sample(num_samples, sigma)

    def pickle_local_model(self, local_model, path):
        """Save the passed local SOGMM model on disk as python pickle.

        Parameters
        ----------
        local_model : gmm_py.GMM4f
            Local SOGMM model to save.
        path : str
            Path for the pickle file.
        """
        if local_model is None:
            print('No model. Model not saved.')
            return

        if path is None:
            path = 'model.pkl'

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def pickle_global_model(self, path=None):
        """Save the current global SOGMM model on disk as python pickle.

        Parameters
        ----------
        path : str, optional (default: None)
            Path for the pickle file.
        """

        if self.model is None:
            print('No model. Model not saved.')
            return

        if path is None:
            path = 'model.pkl'

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_global_model(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


if __name__ == "__main__":
    import os
    import open3d as o3d
    import matplotlib.pyplot as plt
    from termcolor import cprint

    from sogmm_py.utils import ImageUtils, o3d_to_np, calculate_color_metrics

    # load example point clouds and models
    testing_root = '../../test_data/benchmarking'
    pcld_gt_path = os.path.join(testing_root, 'copyroom_000010.pcd')
    model_path = os.path.join(testing_root, 'copyroom_model_000010.pkl')
    copyroom_pose = np.array([[0.9915990557, 0.0438694063,	-0.1216825841,	0.0426045509],
                              [-0.0308488400,	0.9937916320,
                                  0.1068962717, -0.0603890127],
                              [0.1256158249,	-0.1022445597,
                                  0.9867964107,	0.0666403412],
                              [0.0000000000,	0.0000000000,	0.0000000000,	1.0000000000]])

    with open(model_path, 'rb') as mf:
        copyroom_model = pickle.load(mf)

    copyroom_pcd = o3d_to_np(
        o3d.io.read_point_cloud(pcld_gt_path, format='pcd'))
    n_samples = copyroom_pcd.shape[0]

    K = np.eye(3)
    K[0, 0] = 525.0/2
    K[1, 1] = 525.0/2
    K[0, 2] = 319.5/2
    K[1, 2] = 239.5/2
    iu = ImageUtils(K)

    fig, ax = plt.subplots(2, 1)
    d_gt, g_gt = iu.pcld_wf_to_imgs(copyroom_pose, copyroom_pcd)

    sg = SOGMM(0.01)
    sg.model = copyroom_model

    regressed_colors = np.zeros((n_samples, 1))
    regressed_colors = sg.color_conditional(copyroom_pcd[:, 0:3])

    regressed_pcd = np.zeros(copyroom_pcd.shape)
    regressed_pcd[:, 0:3] = copyroom_pcd[:, 0:3]
    regressed_pcd[:, 3] = np.squeeze(regressed_colors)

    _, g_pr = iu.pcld_wf_to_imgs(copyroom_pose, regressed_pcd)

    psnr, ssim = calculate_color_metrics(g_gt, g_pr)

    cprint('psnr %f ssim %f' % (psnr, ssim), 'green')

    o3d.visualization.draw_geometries([np_to_o3d(regressed_pcd)])
