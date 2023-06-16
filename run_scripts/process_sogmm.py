"""Processing output for SOGMM"""

import os
import argparse
import numpy as np
import glob
from termcolor import cprint
import time
import pathlib
import open3d as o3d
import pickle

from sogmm_py.utils import ImageUtils, read_log_trajectory, np_to_o3d
from sogmm_py.vis_open3d import VisOpen3D


def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(1.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback(pcd,
                                                              rotate_view)


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOGMM system")
    parser.add_argument('--bandwidth', type=float)
    parser.add_argument('--decimate', type=int)
    parser.add_argument('--datasetroot', type=dir_path)
    parser.add_argument('--datasetname', type=str)
    parser.add_argument('--resultsroot', type=dir_path)
    parser.add_argument('--colorext', type=str)
    parser.add_argument('--vizcamera', type=bool, default=False)
    args = parser.parse_args()

    bandwidth = args.bandwidth
    decimate_factor = args.decimate
    datasets_root = args.datasetroot
    dataset_name = args.datasetname
    results_root = args.resultsroot

    results_path = os.path.join(results_root, dataset_name + '_' + str(bandwidth))

    pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

    K = np.eye(3)
    K[0, 0] = 525.0/decimate_factor
    K[1, 1] = 525.0/decimate_factor
    K[0, 2] = 319.5/decimate_factor
    K[1, 2] = 239.5/decimate_factor

    # used in open3d visualization
    O3D_K = np.array([[935.30743609,   0.,         959.5],
                      [0.,         935.30743609, 539.5],
                      [0.,           0.,           1.]])

    W = (int)(640/decimate_factor)
    H = (int)(480/decimate_factor)

    rgb_glob_string = os.path.join(
        datasets_root, dataset_name + '-color/*.' + args.colorext)
    depth_glob_string = os.path.join(
        datasets_root, dataset_name + '-depth/*.png')
    traj_string = os.path.join(datasets_root, dataset_name + '-traj.log')

    rgb_paths = sorted(glob.glob(rgb_glob_string))
    depth_paths = sorted(glob.glob(depth_glob_string))
    traj = read_log_trajectory(traj_string)

    iu = ImageUtils(K)
    viz = VisOpen3D()

    full_model_path = os.path.join(results_path, 'full_model.pkl')
    with open(full_model_path, 'rb') as f:
        full_model = pickle.load(f)

    cprint('total number of components in the final model: %d' % (full_model.n_components_), 'cyan')
    cprint('support size of the final model: %d' % (full_model.support_size_), 'cyan')

    local_model_paths = sorted(glob.glob(os.path.join(results_path, 'local_model_*.pkl')))
    frames = [int(os.path.splitext(p)[0].split('/')[-1].split('_')[-1]) for p in local_model_paths]
    for i, fr in enumerate(frames):
        # load the model for this frame
        local_model_path = os.path.join(results_path, 'local_model_' + str(fr) + '.pkl')
        with open(local_model_path, 'rb') as f:
            local_model = pickle.load(f)

        # reconstruct
        pcld_recon = local_model.sample(3 * local_model.support_size_, 2.0)

        # add to the visualizer
        viz.add_geometry(np_to_o3d(pcld_recon))
        if args.vizcamera:
            viz.draw_camera(K, traj[fr].pose, W, H, color=[0.0, 0.0, 0.0])
        viz.update_view_point(O3D_K, np.linalg.inv(traj[fr].pose))
        viz.update_renderer()
        viz.poll_events()

        # load the ground truth for comparison
        pcld_gt, im_gt = iu.generate_pcld_wf(traj[fr].pose, rgb_path=rgb_paths[fr],
                                       depth_path=depth_paths[fr],
                                       size=(W, H))

        cprint('processed frame %d' % (fr), 'grey')

        # adding a sleep just to give the renderer some time to update
        time.sleep(1.0)

    viz.render()