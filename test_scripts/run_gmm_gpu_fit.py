#!/usr/bin/env python3.8
import argparse
import os
import sys
import time
import numpy as np
from termcolor import cprint
import pickle

import open3d as o3d
from sogmm_py.utils import o3d_to_np

from kinit_py import KInitf4CPU
from gmm_open3d_py import GMMf4GPU

def resp_calc(X, N, K, kinit_cpu_stats, stats_file):
    kinit_cpu = KInitf4CPU(K, True, kinit_cpu_stats, stats_file)

    resp_ref = np.zeros((N, K))
    _, indices = kinit_cpu.resp_calc(X)

    resp_ref[indices, np.arange(K)] = 1

    return resp_ref

def fit_gpu(K, X_filepath, model_path, gmm_gpu_stats, kinit_cpu_stats, stats_file):

    # Load the data
    X = o3d_to_np(o3d.io.read_point_cloud(X_filepath))

    # Get number of points
    N = X.shape[0]

    # Create the model
    model = GMMf4GPU(K, N, "CUDA:0", True, gmm_gpu_stats, stats_file)

    # Run KMeans
    start = time.time()
    resp = resp_calc(X, N, K, kinit_cpu_stats, stats_file)
    end = time.time()
    cprint('time taken for CPU C++ kmeans %f seconds' %
           (end - start), 'green')

    # Train the model
    start = time.time()
    success_gpu = model.fit(X, resp)
    end = time.time()
    if success_gpu:
        cprint('Time taken for GPU C++ fit (including data transfers) %f seconds' %
               (end - start), 'green')

        # Dump model to file
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    else:
        cprint('GPU GMM fit has failed.', 'red')
        sys.exit('Exiting')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GMM GPU Fit")
    parser.add_argument('--K', type=int)
    parser.add_argument('--pointcloud', type=str)
    parser.add_argument('--modelpath', type=str)
    parser.add_argument('--gmmgpustats', type=str)
    parser.add_argument('--kinitcpustats', type=str)
    parser.add_argument('--statsfile', type=str)
    args = parser.parse_args()
    fit_gpu(args.K, args.pointcloud, args.modelpath,
            args.gmmgpustats, args.kinitcpustats, args.statsfile)
