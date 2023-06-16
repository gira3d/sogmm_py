#!/usr/bin/env python3.8
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from termcolor import cprint
import pickle
import subprocess
import sys

from tests_utils import *

import open3d as o3d
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture

from sogmm_py.utils import o3d_to_np

from mean_shift_py import MeanShift as MSf2CPU
from kinit_py import KInitf4CPU
from gmm_py import GMMf4CPU


class SOGMMBenchmark:
    def __init__(self, resultsroot, decimate):
        self.results_root = resultsroot
        self.decimate = decimate

        self.ms_cpu_stats = make_dir(
            self.results_root, 'ms_cpu_stats_' + str(decimate).replace(".", "_"))
        self.kinit_cpu_stats = make_dir(
            self.results_root, 'kinit_cpu_stats_' + str(decimate).replace(".", "_"))
        self.ms_cpu_models = make_dir(
            self.results_root, 'ms_cpu_models_' + str(decimate).replace(".", "_"))
        self.kinit_cpu_models = make_dir(
            self.results_root, 'kinit_cpu_models_' + str(decimate).replace(".", "_"))
        self.gmm_cpu_models = make_dir(
            self.results_root, 'gmm_cpu_models_' + str(decimate).replace(".", "_"))
        self.gmm_cpu_stats = make_dir(
            self.results_root, 'gmm_cpu_stats_' + str(decimate).replace(".", "_"))

        self.gmm_gpu_stats = make_dir(
            self.results_root, 'gmm_gpu_stats_' + str(decimate).replace(".", "_"))
        self.gmm_gpu_models = make_dir(
            self.results_root, 'gmm_gpu_models_' + str(decimate).replace(".", "_"))

        self.ms_scikit_stats = make_dir(
            self.results_root, 'ms_scikit_stats_' + str(decimate).replace(".", "_"))
        self.kinit_scikit_stats = make_dir(
            self.results_root, 'kinit_scikit_stats_' + str(decimate).replace(".", "_"))
        self.gmm_scikit_stats = make_dir(
            self.results_root, 'gmm_scikit_stats_' + str(decimate).replace(".", "_"))

    def compute_n_components(self, X, b, stats_file):
        d = np.array([np.linalg.norm(x) for x in X[:, 0:3]])[:, np.newaxis]
        g = X[:, 3][:, np.newaxis]
        ms_data = np.concatenate((d, g), axis=1)
        mean_shift = MSf2CPU(b, True, self.ms_cpu_stats, stats_file)
        start = time.time()
        mean_shift.fit(ms_data)
        end = time.time()
        cprint('time taken for CPU C++ mean shift %f seconds' %
               (end - start), 'green')
        return mean_shift.get_num_modes()

    def compute_n_components_scikit(self, X, b, stats_file):
        d = np.array([np.linalg.norm(x) for x in X[:, 0:3]])[:, np.newaxis]
        g = X[:, 3][:, np.newaxis]
        ms_data = np.concatenate((d, g), axis=1)
        mean_shift = MeanShift(bandwidth=b, bin_seeding=True)
        start = time.time()
        mean_shift.fit(ms_data)
        lbls = mean_shift.labels_
        ret = len(np.unique(lbls))
        end = time.time()
        cprint('time taken for CPU Python mean shift %f seconds' %
               (end - start), 'green')
        df = pd.DataFrame({'Key': ['fit'],
                            'Sum': [end-start]})
        df.set_index('Key')
        df.to_csv(os.path.join(self.ms_scikit_stats, stats_file))
        return ret

    def resp_calc(self, X, N, K, stats_file):
        kinit_cpu = KInitf4CPU(K, True, self.kinit_cpu_stats, stats_file)

        resp_ref = np.zeros((N, K))
        _, indices = kinit_cpu.resp_calc(X)

        resp_ref[indices, np.arange(K)] = 1

        return resp_ref

    def resp_calc_scikit(self, X, N, K, stats_file):
        resp_ref = np.zeros((N, K))
        start = time.time()
        _, indices = kmeans_plusplus(
            X,
            K,
            random_state=0,
        )
        end = time.time()
        cprint('time taken for CPU Python KInit %f seconds' %
               (end - start), 'green')

        df = pd.DataFrame({'Key': ['respCalc'],
                            'Sum': [end-start]})
        df.set_index('Key')
        df.to_csv(os.path.join(self.kinit_scikit_stats, stats_file))

        resp_ref[indices, np.arange(K)] = 1

        return resp_ref

    def fit_cpu_scikit(self, X, b, stats_file):
        N = X.shape[0]

        K = self.compute_n_components_scikit(X, b, stats_file)
        cprint('number of estimated components: %d' %
               (K), 'cyan')

        gmm_cpu = GaussianMixture(
            n_components=K, init_params='k-means++', random_state=0)

        resp = self.resp_calc_scikit(X, N, K, stats_file)

        start = time.time()
        success_cpu = gmm_cpu.fit(X, resp)
        end = time.time()
        if success_cpu:
            cprint('Time taken for CPU C++ fit %f seconds' %
                   (end - start), 'green')
            df = pd.DataFrame({'Key': ['fit'],
                               'Sum': [end-start]})
            df.set_index('Key')
            df.to_csv(os.path.join(self.gmm_scikit_stats, stats_file))
            return gmm_cpu, K
        else:
            cprint('CPU GMM fit has failed.', 'red')
            sys.exit('Exiting at bandwidth = %f' % (b))

    def fit_cpu(self, X, b, stats_file):
        N = X.shape[0]

        K = self.compute_n_components(X, b, stats_file)
        cprint('number of estimated components: %d' %
               (K), 'cyan')

        gmm_cpu = GMMf4CPU(K, True, self.gmm_cpu_stats, stats_file)

        start = time.time()
        resp = self.resp_calc(X, N, K, stats_file)
        end = time.time()
        cprint('time taken for CPU C++ kmeans %f seconds' %
               (end - start), 'green')

        start = time.time()
        success_cpu = gmm_cpu.fit(X, resp)
        end = time.time()
        if success_cpu:
            cprint('Time taken for CPU C++ fit %f seconds' %
                   (end - start), 'green')
            return gmm_cpu, K
        else:
            cprint('CPU GMM fit has failed.', 'red')
            sys.exit('Exiting at bandwidth = %f' % (b))

    def gpu_bandwidth_benchmark(self, X_filepath, st_bandwidth=0.0135, en_bandwidth=0.03, sep=10, trials=1):
        X = o3d_to_np(o3d.io.read_point_cloud(X_filepath))
        bandwidths = np.linspace(st_bandwidth, en_bandwidth, sep)

        comps = []
        for b in bandwidths:
            for trial in range(trials):
                cprint('running fit test using bandwidth = %f' % b, 'yellow')

                stats_file = 'stats_' + str(b) + '_' + str(trial) + '.csv'

                K = self.compute_n_components(X, b, stats_file)
                cprint('number of estimated components: %d' %
                       (K), 'cyan')
                comps.append(K)

                model_path = os.path.join(
                    self.gmm_gpu_models, 'model_' + str(b) + '_' + str(trial) + '.pkl')

                command = 'python run_gmm_gpu_fit.py --K ' + str(K) + ' --pointcloud ' + X_filepath + ' --modelpath ' + model_path + \
                    ' --gmmgpustats ' + self.gmm_gpu_stats + ' --kinitcpustats ' + \
                    self.kinit_cpu_stats + ' --statsfile ' + stats_file
                subprocess.call(command, shell=True)

        comps = np.array(comps)

        comps_path = os.path.join(self.gmm_gpu_stats, 'comps.csv')
        bwidths_path = os.path.join(self.gmm_gpu_stats, 'bandwidths.csv')

        np.savetxt(comps_path, comps)
        np.savetxt(bwidths_path, bandwidths)

    def cpu_scikit_bandwidth_benchmark(self, X, st_bandwidth=0.0135, en_bandwidth=0.03, sep=10, trials=1):
        bandwidths = np.linspace(st_bandwidth, en_bandwidth, sep)

        comps = []
        for b in bandwidths:
            for trial in range(trials):
                cprint('running fit test using bandwidth = %f' % b, 'yellow')

                stats_file = 'stats_' + str(b) + '_' + str(trial) + '.csv'

                model, K = self.fit_cpu_scikit(X, b, stats_file)

                model_path = os.path.join(
                    self.gmm_cpu_models, 'model_' + str(b) + '_' + str(trial) + '.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

                comps.append(K)

        comps = np.array(comps)

        comps_path = os.path.join(self.gmm_cpu_stats, 'comps.csv')
        bwidths_path = os.path.join(self.gmm_cpu_stats, 'bandwidths.csv')

        np.savetxt(comps_path, comps)
        np.savetxt(bwidths_path, bandwidths)

    def cpu_bandwidth_benchmark(self, X, st_bandwidth=0.0135, en_bandwidth=0.03, sep=10, trials=1):
        bandwidths = np.linspace(st_bandwidth, en_bandwidth, sep)

        comps = []
        for b in bandwidths:
            for trial in range(trials):
                cprint('running fit test using bandwidth = %f' % b, 'yellow')

                stats_file = 'stats_' + str(b) + '_' + str(trial) + '.csv'

                model, K = self.fit_cpu(X, b, stats_file)

                model_path = os.path.join(
                    self.gmm_cpu_models, 'model_' + str(b) + '_' + str(trial) + '.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

                comps.append(K)

        comps = np.array(comps)

        comps_path = os.path.join(self.gmm_cpu_stats, 'comps.csv')
        bwidths_path = os.path.join(self.gmm_cpu_stats, 'bandwidths.csv')

        np.savetxt(comps_path, comps)
        np.savetxt(bwidths_path, bandwidths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOGMM system")
    parser.add_argument('--resultsroot', type=dir_path)
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--cpu', type=bool, default=False)
    parser.add_argument('--scikit', type=bool, default=False)
    parser.add_argument('--decimate', type=float, default=2)
    args = parser.parse_args()

    tester = SOGMMBenchmark(args.resultsroot, args.decimate)

    X_filepath = 'pcd_854_decimate_' + \
        str(args.decimate).replace(".", "_") + '.pcd'

    if args.gpu:
        tester.gpu_bandwidth_benchmark(X_filepath, sep=10, trials=10)

    if args.cpu:
        X = o3d_to_np(o3d.io.read_point_cloud(X_filepath))
        tester.cpu_bandwidth_benchmark(X, sep=10, trials=10)

    if args.scikit:
        X = o3d_to_np(o3d.io.read_point_cloud(X_filepath))
        tester.cpu_scikit_bandwidth_benchmark(X, sep=10, trials=10)
