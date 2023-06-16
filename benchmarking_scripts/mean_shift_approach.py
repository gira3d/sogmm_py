import os
import time
import argparse
import numpy as np
from termcolor import cprint
import matplotlib.pyplot as plt
import tikzplotlib as tz
import pprint
import pickle

from benchmark_utils import DatasetProcessor
from mean_shift_py import MeanShift as MSf2
from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import *

pp = pprint.PrettyPrinter()


def mean_shift(pcld, b):
    d = np.array([np.linalg.norm(x) for x in pcld[:, 0:3]])[:, np.newaxis]
    g = pcld[:, 3][:, np.newaxis]
    ms_data = np.concatenate((d, g), axis=1)

    ms = MSf2(b)
    s = time.time()
    ms.fit(ms_data)
    e = time.time()
    cprint('time taken for bandwidth %f is %f' % (b, e - s), 'green')

    return ms.get_num_modes(), e - s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mean shift approach py")
    parser.add_argument('--datasetname', type=str)
    parser.add_argument('--datasetroot', type=str)
    parser.add_argument('--resultsroot', type=str)
    parser.add_argument('--decimate', type=int)
    parser.add_argument('--frames', nargs='+', type=str)
    parser.add_argument('--tikz', type=bool)

    args = parser.parse_args()

    bandwidths = np.linspace(0.01, 0.05, 10)
    cprint(bandwidths, 'red')

    dataset_path = os.path.join(args.datasetroot, args.datasetname)

    data_process = DatasetProcessor(datasets_root=args.datasetroot,
                                    dataset_name=args.datasetname,
                                    results_root=args.resultsroot,
                                    decimate_factor=args.decimate)

    comps = {}
    times = {}
    recon_errs = {}
    colors = ['red', 'blue']
    labels = ['Image B', 'Image A']
    comps_fig, comps_ax = plt.subplots()
    comps_ax.set_xlim([min(bandwidths), max(bandwidths)])
    recon_err_fig, recon_err_ax = plt.subplots()
    recon_err_ax.set_xlim([min(bandwidths), max(bandwidths)])
    times_fig, times_ax = plt.subplots()
    times_ax.set_xlim([min(bandwidths), max(bandwidths)])
    for i, f in enumerate(args.frames):
        cprint('processing frame %s in dataset %s' %
               (f, args.datasetname), 'grey')
        f_comps = []
        dt_comps = []
        f_recon_errs = []

        pcld, imgs = data_process.get_pcd_and_images(f, rgb_ext='.jpg')
        gt_pose = data_process.get_gt_pose(int(f))
        gt_pcd = data_process.get_local_gt_pcd(int(f))
        for b in bandwidths:
            c, dt = mean_shift(pcld, b)
            f_comps.append(c)
            dt_comps.append(dt)

            print(args.datasetname + '_' + str(b))
            results_path = os.path.join(args.resultsroot, args.datasetname +
                                        '_sogmm/' + args.datasetname + '_' + str(b))
            model_path = os.path.join(
                results_path, 'local_model_' + str(int(f)) + '.pkl')

            with open(model_path, 'rb') as mf:
                model = pickle.load(mf)

            assert (model.support_size_ == gt_pcd.shape[0])

            # evaluate recon. error, precision, recall, f-score
            n_samples = gt_pcd.shape[0]
            sg = SOGMM()
            sg.model = model
            pr_pcd = sg.joint_dist_sample(n_samples)
            fs, p, re, rmean, rstd = calculate_depth_metrics(
                np_to_o3d(gt_pcd), np_to_o3d(pr_pcd))

            print(rmean)
            f_recon_errs.append(rmean)

        comps[f] = f_comps
        times[f] = dt_comps
        recon_errs[f] = f_recon_errs

        comps_ax.plot(bandwidths, comps[f], marker='s',
                      color=colors[i], label=labels[i])
        recon_err_ax.plot(bandwidths, recon_errs[f], marker='o',
                      color=colors[i], label=labels[i])
        times_ax.plot(bandwidths, times[f], marker='s',
                      color=colors[i], label=labels[i])

    # # comps_ax.legend()
    comps_ax.set_xlabel('Bandwidth')
    comps_ax.set_ylabel('$M$')

    if args.tikz:
        tz.save('mean_shift_approach_fig.tex')

    # recon_err_ax.legend()
    recon_err_ax.set_ylabel('Recon. Err.')
    recon_err_ax.set_xlabel('Bandwidth')

    if args.tikz:
        tz.save('recon_err_fig.tex', figure=recon_err_fig)

    # times_ax.legend()
    times_ax.set_ylim([0, 2.0])
    times_ax.set_xlabel('Bandwidth')
    times_ax.set_ylabel('PRI Time (s)')

    if args.tikz:
        tz.save('mean_shift_approach_times.tex')

    # plt.show()
