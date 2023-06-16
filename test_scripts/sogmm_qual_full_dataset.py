#!/usr/bin/env python3.8
import argparse
import os
from termcolor import cprint
import subprocess
import open3d as o3d
import numpy as np
import pandas as pd
import pickle

from tests_utils import *
from sogmm_py.utils import o3d_to_np, read_log_trajectory, ImageUtils, calculate_color_metrics, calculate_depth_metrics, np_to_o3d
from sogmm_benchmark import SOGMMBenchmark
from sogmm_py import SOGMM


def evaluate_gmm_model(model, gt_pose, gt_pcd, iu):
    n_samples = gt_pcd.shape[0]
    _, gt_g = iu.pcld_wf_to_imgs(gt_pose, gt_pcd)

    sg = SOGMM()
    sg.model = model

    # evaluate recon. error, precision, recall, f-score
    pr_pcd = sg.joint_dist_sample(n_samples)
    f, p, re, rmean, rstd = calculate_depth_metrics(
        np_to_o3d(gt_pcd), np_to_o3d(pr_pcd))

    # evaluate PSNR and SSIM
    regressed_colors = np.zeros((n_samples, 1))
    regressed_colors = sg.color_conditional(gt_pcd[:, 0:3])

    regressed_pcd = np.zeros(gt_pcd.shape)
    regressed_pcd[:, 0:3] = gt_pcd[:, 0:3]
    regressed_pcd[:, 3] = np.squeeze(regressed_colors)

    _, pr_g = iu.pcld_wf_to_imgs(gt_pose, regressed_pcd)

    gt_g = np.nan_to_num(gt_g)
    pr_g = np.nan_to_num(pr_g)

    psnr, ssim = calculate_color_metrics(gt_g, pr_g)

    # memory
    M = model.n_components_
    mem_bytes = 4 * M * (1 + 10 + 4)

    return psnr, rmean, mem_bytes, pr_pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOGMM qual")
    parser.add_argument('--resultsroot', type=dir_path)
    parser.add_argument('--decimate', nargs='+', type=float)
    parser.add_argument('--nframes', type=int)
    parser.add_argument('--bandwidths', nargs='+', type=float)
    parser.add_argument('--datasetroot', type=dir_path)
    parser.add_argument('--datasetname', type=str)
    args = parser.parse_args()

    # get the ground truth trajectory
    traj_string = os.path.join(
        args.datasetroot, args.datasetname + '-traj.log')
    traj = read_log_trajectory(traj_string)

    # declare the common intrinsic matrix at full resolution
    K = np.eye(3)
    K[0, 0] = 525.0
    K[1, 1] = 525.0
    K[0, 2] = 319.5
    K[1, 2] = 239.5

    # image utils object used for full resolution image
    iu = ImageUtils(K)

    # process up to the given number of frames
    for f in range(args.nframes):
        # get the ground truth undecimated point cloud
        pcld_gt = os.path.join(args.datasetroot, 'pcd_' + str(args.datasetname) + '_' + \
            str(f) + '_decimate_1_0.pcd')
        X_gt = o3d_to_np(o3d.io.read_point_cloud(pcld_gt))
        N_gt = X_gt.shape[0]

        # learn the model for each given decimation
        for d in args.decimate:
            # get the decimated point cloud over which the model will be learnt
            X_filepath = os.path.join(args.datasetroot, 'pcd_' + str(args.datasetname) + '_' + str(f) + '_decimate_' + \
                str(d).replace(".", "_") + '.pcd')
            X = o3d_to_np(o3d.io.read_point_cloud(X_filepath))

            tester = SOGMMBenchmark(args.resultsroot, d)

            for b in args.bandwidths:
                stats_file = 'stats_' + str(args.datasetname) + '_' + str(f) + '_' + str(d).replace(".", "_") + \
                    '_' + str(b).replace(".", "_") + '.csv'

                K = tester.compute_n_components(X, b, stats_file)
                cprint('number of estimated components: %d' % (K), 'cyan')

                model_path = os.path.join(
                    tester.gmm_gpu_models, 'model_' + str(args.datasetname) + '_' + str(f) + '_' + str(d).replace(".", "_") + '_' + str(b).replace(".", "_") + '.pkl')

                command = 'python run_gmm_gpu_fit.py --K ' + str(K) + ' --pointcloud ' + X_filepath + ' --modelpath ' + model_path + \
                    ' --gmmgpustats ' + tester.gmm_gpu_stats + ' --kinitcpustats ' + \
                    tester.kinit_cpu_stats + ' --statsfile ' + stats_file
                subprocess.call(command, shell=True)

            metrics = np.zeros((len(args.bandwidths), 6))
            metrics_file = 'metrics_' + str(args.datasetname) + '_' + str(f) + '_' + str(d).replace(".", "_") + '.csv'
            for j, b in enumerate(args.bandwidths):
                model_path = os.path.join(
                    tester.gmm_gpu_models, 'model_' + str(args.datasetname) + '_' + str(f) + '_' + str(d).replace(".", "_") + '_' + str(b).replace(".", "_") + '.pkl')

                if not os.path.isfile(model_path):
                    continue

                with open(model_path, 'rb') as fl:
                    model = pickle.load(fl)

                psnr, rmean, mem, X_pr = evaluate_gmm_model(
                    model, traj[f].pose, X_gt, iu)

                cprint('psnr %f support size %d' %
                       (psnr, model.support_size_), 'green')

                metrics[j, 0] = b
                metrics[j, 1] = model.n_components_
                metrics[j, 2] = model.support_size_
                metrics[j, 3] = psnr
                metrics[j, 4] = rmean
                metrics[j, 5] = mem

                recon_file = 'recon_' + str(args.datasetname) + '_' + str(f) + '_' + str(d).replace(".", "_") + \
                    '_' + str(b).replace(".", "_") + '.pcd'
                recon_path = os.path.join(args.resultsroot, recon_file)

                o3d.io.write_point_cloud(recon_path, np_to_o3d(X_pr))

            df = pd.DataFrame(metrics, columns=[
                              'bandwidth', 'comps', 'support', 'psnr', 'rmean', 'mem'])
            df.set_index('bandwidth')
            df.to_csv(os.path.join(args.resultsroot, metrics_file))
