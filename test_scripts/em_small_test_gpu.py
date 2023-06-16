#!/usr/bin/env python3.8
import time
import sys
import argparse
import numpy as np
from termcolor import cprint

from tests_utils import *

import open3d as o3d
from sogmm_py.utils import o3d_to_np

from kinit_py import KInitf2CPU, KInitf3CPU, KInitf4CPU
from gmm_py import GMMf2CPU, GMMf3CPU, GMMf4CPU
from gmm_open3d_py import GMMf2GPU, GMMf3GPU, GMMf4GPU


class EMSmallTestGPU:
    def __init__(self, X, K, D):
        # dataset
        self.X = X

        # dimensions
        self.n_samples = self.X.shape[0]
        self.n_components = K
        self.dim = D

        # responsibility matrix
        self.resp_ref = np.zeros((self.n_samples, self.n_components))

        if self.dim == 2:
            self.kinit_cpu = KInitf2CPU(self.n_components)
            self.gmm_gpu = GMMf2GPU(self.n_components, self.n_samples, "CUDA:0")
            self.gmm_cpu = GMMf2CPU(self.n_components, 1e-3, 1e-6, 100)
        elif self.dim == 3:
            self.kinit_cpu = KInitf3CPU(self.n_components)
            self.gmm_gpu = GMMf3GPU(self.n_components, self.n_samples, "CUDA:0")
            self.gmm_cpu = GMMf3CPU(self.n_components, 1e-3, 1e-6, 100)
        elif self.dim == 4:
            self.kinit_cpu = KInitf4CPU(self.n_components)
            self.gmm_gpu = GMMf4GPU(self.n_components, self.n_samples, "CUDA:0")
            self.gmm_cpu = GMMf4CPU(self.n_components, 1e-3, 1e-6, 100)
        else:
            sys.exit("Only dimensions 2, 3, and 4 are allowed.")

    def run_kmeans(self):
        _, indices = self.kinit_cpu.resp_calc(self.X)
        self.resp_ref[indices, np.arange(self.n_components)] = 1

    def test_m_step(self):
        success_cpu = self.gmm_cpu.fit(self.X, self.resp_ref)

        if success_cpu:
            resp_ref = self.gmm_cpu.resp_
            self.gmm_gpu.m_step(self.X, resp_ref)

            w = self.gmm_gpu.weights_
            m = self.gmm_gpu.means_
            C = self.gmm_gpu.covariances_

            np.testing.assert_array_almost_equal(
                w, self.gmm_cpu.weights_, decimal=4)
            np.testing.assert_array_almost_equal(
                m, self.gmm_cpu.means_, decimal=4)
            np.testing.assert_array_almost_equal(
                C, self.gmm_cpu.covariances_, decimal=4)
        else:
            cprint('m step test requires a CPU fit', 'red')
            cprint('CPU fit did not succeed', 'red')

    def test_e_step(self):
        self.gmm_gpu.update_device_and_host_external(self.gmm_cpu.weights_,
                                                     self.gmm_cpu.means_,
                                                     self.gmm_cpu.covariances_,
                                                     self.gmm_cpu.precisions_cholesky_)
        log_resp_gpu = self.gmm_gpu.e_step(self.X)
        _, log_resp_cpu = self.gmm_cpu.e_step(self.X)
        np.testing.assert_array_almost_equal(
            log_resp_gpu, log_resp_cpu, decimal=3)

    def test_fit(self):
        start = time.time()
        self.run_kmeans()
        end = time.time()
        cprint('time taken for CPU C++ kmeans %f seconds' %
               (end - start), 'red')

        start = time.time()
        success_gpu = self.gmm_gpu.fit(self.X, self.resp_ref)
        end = time.time()
        cprint('time taken for GPU C++ fit (including data transfers) %f seconds' %
               (end - start), 'green')

        start = time.time()
        success_cpu = self.gmm_cpu.fit(self.X, self.resp_ref)
        end = time.time()
        cprint('time taken for Eigen CPU C++ fit %f seconds' %
               (end - start), 'red')

        if success_gpu and success_cpu:
            cprint('Fits succeeded.', 'green')
            np.testing.assert_array_almost_equal(
                self.gmm_gpu.weights_, self.gmm_cpu.weights_, decimal=3)
            np.testing.assert_array_almost_equal(
                self.gmm_gpu.means_, self.gmm_cpu.means_, decimal=3)
            np.testing.assert_array_almost_equal(
                self.gmm_gpu.covariances_, self.gmm_cpu.covariances_, decimal=3)
            cprint('All tests passed.', 'green')
        else:
            cprint('One or both of the fits failed.', 'red')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EM Small Test GPU")
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--components', type=int, default=2)
    parser.add_argument('--dim', type=int, default=2)
    args = parser.parse_args()

    X = o3d_to_np(o3d.io.read_point_cloud('pcld_gt.pcd'))
    # X = np.random.rand(args.samples, args.dim)

    tester = EMSmallTestGPU(X, args.components, args.dim)

    cprint('running fit test on im data', 'yellow')
    tester.test_fit()
