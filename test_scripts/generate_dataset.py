import os
import argparse
import numpy as np
import glob
from termcolor import cprint
import open3d as o3d

from sogmm_py.utils import ImageUtils, read_log_trajectory, np_to_o3d


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOGMM system")
    parser.add_argument('--decimate', nargs='+', type=float)
    parser.add_argument('--datasetroot', type=dir_path)
    parser.add_argument('--nframes', type=int)
    parser.add_argument('--destroot', type=dir_path)
    parser.add_argument('--datasetname', type=str)
    parser.add_argument('--colorext', type=str)
    args = parser.parse_args()

    decimate_factors = args.decimate
    datasets_root = args.datasetroot
    dest_root = args.destroot
    dataset_name = args.datasetname

    rgb_glob_string = os.path.join(
        datasets_root, dataset_name + '-color/*.' + args.colorext)
    depth_glob_string = os.path.join(
        datasets_root, dataset_name + '-depth/*.png')
    traj_string = os.path.join(datasets_root, dataset_name + '-traj.log')

    rgb_paths = sorted(glob.glob(rgb_glob_string))
    depth_paths = sorted(glob.glob(depth_glob_string))
    traj = read_log_trajectory(traj_string)

    n_frames = len(rgb_paths)

    if args.nframes < 0:
        # use all frames
        args.nframes = n_frames

    assert (len(depth_paths) == n_frames)
    assert (len(traj) == n_frames)

    for decimate_factor in decimate_factors:
        K = np.eye(3)
        K[0, 0] = 525.0/decimate_factor
        K[1, 1] = 525.0/decimate_factor
        K[0, 2] = 319.5/decimate_factor
        K[1, 2] = 239.5/decimate_factor

        W = (int)(640/decimate_factor)
        H = (int)(480/decimate_factor)

        iu = ImageUtils(K)

        for i, fr in enumerate(range(args.nframes)):
            pcd_path = os.path.join(dest_root, 'pcd_' + str(args.datasetname) + '_' + str(
                fr) + '_decimate_' + str(decimate_factor).replace(".", "_") + '.pcd')
            pcld_gt, im = iu.generate_pcld_wf(traj[fr].pose, rgb_path=rgb_paths[fr],
                                              depth_path=depth_paths[fr],
                                              size=(W, H))
            o3d.io.write_point_cloud(pcd_path, np_to_o3d(pcld_gt))
            cprint('generated %s' % (pcd_path), 'green')
