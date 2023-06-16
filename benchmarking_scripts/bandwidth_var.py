import os
import pathlib
import argparse
import glob
import numpy as np
from termcolor import cprint
import open3d as o3d

from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import ImageUtils, read_log_trajectory, np_to_o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mean shift approach py")
    parser.add_argument('--datasetname', type=str)
    parser.add_argument('--datasetroot', type=str)
    parser.add_argument('--resultsroot', type=str)
    parser.add_argument('--decimate', type=int)
    parser.add_argument('--frames', nargs='+', type=int)
    parser.add_argument('--tikz', type=bool)

    args = parser.parse_args()

    decimate_factor = args.decimate
    datasets_root = args.datasetroot
    dataset_name = args.datasetname
    results_root = args.resultsroot

    bandwidths = np.linspace(0.01, 0.05, 10)
    cprint(bandwidths, 'red')

    K = np.eye(3)
    K[0, 0] = 525.0/decimate_factor
    K[1, 1] = 525.0/decimate_factor
    K[0, 2] = 319.5/decimate_factor
    K[1, 2] = 239.5/decimate_factor

    W = (int)(640/decimate_factor)
    H = (int)(480/decimate_factor)

    iu = ImageUtils(K)

    rgb_glob_string = os.path.join(
        datasets_root, dataset_name + '-color/*.jpg')
    cprint('rgb glob string %s' % (rgb_glob_string), 'green')
    depth_glob_string = os.path.join(
        datasets_root, dataset_name + '-depth/*.png')
    cprint('depth glob string %s' % (depth_glob_string), 'green')
    traj_string = os.path.join(datasets_root, dataset_name + '-traj.log')
    cprint('traj path %s' % (traj_string), 'green')

    rgb_paths = sorted(glob.glob(rgb_glob_string))
    depth_paths = sorted(glob.glob(depth_glob_string))
    traj = read_log_trajectory(traj_string)

    for i, fr in enumerate(args.frames):
        cprint('processing frame %s in dataset %s' % (fr, args.datasetname), 'grey')

        for b in bandwidths:
            results_path = os.path.join(results_root, dataset_name + '_sogmm/' + dataset_name + '_' + str(b))

            local_pcd_path = os.path.join(results_path, 'local_pcd_' + str(fr) + '.pcd')
            pcld_gt, _ = iu.generate_pcld_wf(traj[fr].pose, rgb_path=rgb_paths[fr],
                                        depth_path=depth_paths[fr],
                                        size=(W, H))
            pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

            sg = SOGMM(b)
            local_model_path = os.path.join(results_path, 'local_model_' + str(fr) + '.pkl')
            local_model = sg.fit(pcld_gt)
            sg.pickle_local_model(local_model, local_model_path)
            o3d.io.write_point_cloud(local_pcd_path, np_to_o3d(pcld_gt))