import argparse
import os
import numpy as np
import glob
from cprint import *
import copy

from nigh_py import R3KDTree, SO3KDTree

from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import *


class ISOGMMFullFrame:
    def __init__(self, path_datasets, dataset_name, nframes=None, bandwidth=0.025, deci=8.0, compute='GPU'):
        # paths to all rgb and depth images in the dataset
        self.rgb_paths = sorted(
            glob.glob(os.path.join(path_datasets, dataset_name + '-color/*.jpg')))
        self.depth_paths = sorted(
            glob.glob(os.path.join(path_datasets, dataset_name + '-depth/*.png')))

        # read the full camera trajectory
        self.traj = read_log_trajectory(os.path.join(
            path_datasets, dataset_name + '-traj.log'))

        # image/pcld utility
        self.K_d = np.eye(3)
        self.K_d[0, 0] = 525.0/deci
        self.K_d[1, 1] = 525.0/deci
        self.K_d[0, 2] = 319.5/deci
        self.K_d[1, 2] = 239.5/deci
        self.W_d = (int)(640/deci)
        self.H_d = (int)(480/deci)
        self.iu_d = ImageUtils(self.K_d, im_h=self.H_d, im_w=self.W_d)

        # SOGMM class object initialize
        self.bandwidth = bandwidth
        self.compute = compute
        self.sg = SOGMM(self.bandwidth, compute=self.compute)

        # make sure the number of rgb and depth images are the same
        assert (len(self.rgb_paths) == len(self.depth_paths))
        if nframes is None:
            self.nframes = len(self.rgb_paths)
        else:
            self.nframes = nframes

        # other parameters
        self.sensor_speed = 1.0
        self.frame_rate = 10

        # initialize KDTrees
        # r3_radius = self.sensor_speed * (1.0 / self.frame_rate)
        r3_radius = 0.10
        so3_radius = np.pi / 2.0
        self.r3_tree = R3KDTree(r3_radius)
        self.so3_tree = SO3KDTree(so3_radius)

    def process_frame(self, n):
        cprint.ok(f'frame {n}')

        if self.sg.model is not None:
            cprint.info(
                f'number of components in the global GMM model: {self.sg.model.n_components_}')

        # sensor information
        self.pose = self.traj[n].pose
        t = self.pose[:3, 3]
        R = self.pose[:3, :3]

        # point cloud information
        pcld, self.im = self.iu_d.generate_pcld_wf(self.pose, rgb_path=self.rgb_paths[n],
                                                   depth_path=self.depth_paths[n],
                                                   size=(self.W_d, self.H_d))

        learn_model = False
        if self.sg.model is None:
            # learn it
            learn_model = True
        else:
            # find the closest poses in R3 and SO3
            r3_out = self.r3_tree.search(t)
            so3_out = self.so3_tree.search(R)

            if len(r3_out) == 0 and len(so3_out):
                cprint.ok(f'Frame {n} is novel.')
                learn_model = True
            else:
                learn_model = False

        if learn_model:
            cprint.info('Fitting Model.')
            # fit
            self.sg.fit(pcld)
            # add to the KDTrees
            self.r3_tree.insert(t)
            self.so3_tree.insert(R.flatten())
            # return True to indicate that the model was updated
            return True
        else:
            cprint.warn(f'Skipping frame {n}.')
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Incremental-SOGMM")
    parser.add_argument('--path_datasets', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--nframes', type=int)
    parser.add_argument('--bandwidth', type=float)

    args = parser.parse_args()

    isogmm = ISOGMMFullFrame(args.path_datasets,
                             args.dataset_name,
                             nframes=args.nframes,
                             bandwidth=args.bandwidth)

    isogmm.process_frame(0)
    isogmm.process_frame(1)
    isogmm.process_frame(2)
