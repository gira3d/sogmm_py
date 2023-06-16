"""Main running script for SOGMM"""

import os
import pathlib
import argparse
import numpy as np
import glob
from termcolor import cprint
import time
import open3d as o3d

from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import ImageUtils, read_log_trajectory, np_to_o3d


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOGMM system")
    parser.add_argument('--bandwidth', type=float)
    parser.add_argument('--numcomponents', type=int, default=0)
    parser.add_argument('--decimate', type=int)
    parser.add_argument('--datasetroot', type=dir_path)
    parser.add_argument('--datasetname', type=str)
    parser.add_argument('--colorext', type=str)
    parser.add_argument('--resultsroot', type=dir_path)
    parser.add_argument('--frames', nargs='+', type=int)
    args = parser.parse_args()

    fixed_num_components = False
    bandwidth = args.bandwidth
    n_comps = args.numcomponents
    # to force this script to use fixed number of components, pass negative
    # number in bandwidth argument
    if bandwidth < 0 and n_comps > 1:
        cprint('fixed components case with %d components' % (n_comps), 'red')
        fixed_num_components = True

    decimate_factor = args.decimate
    datasets_root = args.datasetroot
    dataset_name = args.datasetname
    results_root = args.resultsroot

    if fixed_num_components:
        results_path = os.path.join(results_root, dataset_name + '_fixed_comps')
    else:
        results_path = os.path.join(results_root, dataset_name + '_sogmm/' + dataset_name + '_' + str(bandwidth))

    pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

    K = np.eye(3)
    K[0, 0] = 525.0/decimate_factor
    K[1, 1] = 525.0/decimate_factor
    K[0, 2] = 319.5/decimate_factor
    K[1, 2] = 239.5/decimate_factor

    W = (int)(640/decimate_factor)
    H = (int)(480/decimate_factor)

    rgb_glob_string = os.path.join(
        datasets_root, dataset_name + '-color/*.' + args.colorext)
    cprint('rgb glob string %s' % (rgb_glob_string), 'green')
    depth_glob_string = os.path.join(
        datasets_root, dataset_name + '-depth/*.png')
    cprint('depth glob string %s' % (depth_glob_string), 'green')
    traj_string = os.path.join(datasets_root, dataset_name + '-traj.log')
    cprint('traj path %s' % (traj_string), 'green')

    rgb_paths = sorted(glob.glob(rgb_glob_string))
    depth_paths = sorted(glob.glob(depth_glob_string))
    traj = read_log_trajectory(traj_string)

    n_frames = len(rgb_paths)

    print(len(depth_paths), n_frames)
    assert(len(depth_paths) == n_frames)
    assert(len(traj) == n_frames)

    sg = SOGMM(bandwidth)
    iu = ImageUtils(K)

    frames = args.frames
    for i, fr in enumerate(frames):
        local_pcd_path = os.path.join(results_path, 'local_pcd_' + str(fr) + '.pcd')
        pcld_gt, im = iu.generate_pcld_wf(traj[fr].pose, rgb_path=rgb_paths[fr],
                                       depth_path=depth_paths[fr],
                                       size=(W, H))
        start = time.time()
        if not fixed_num_components:
            local_model_path = os.path.join(results_path, 'local_model_' + str(fr) + '.pkl')
            local_model = sg.fit(pcld_gt)
        else:
            local_model_path = os.path.join(results_path, 'local_model_' + str(n_comps) + '_' + str(fr) + '.pkl')
            local_model = sg.gmm_fit(pcld_gt, n_comps)
        end = time.time()
        cprint('processed frame %d' % (fr), 'grey')
        cprint('time taken %f seconds' % (end - start), 'green')
        sg.pickle_local_model(local_model, local_model_path)
        o3d.io.write_point_cloud(local_pcd_path, np_to_o3d(pcld_gt))

    if not fixed_num_components:
        sg.pickle_global_model(os.path.join(results_path, 'full_model.pkl'))
    else:
        sg.pickle_global_model(os.path.join(results_path, 'full_model_' + str(n_comps) + '.pkl'))