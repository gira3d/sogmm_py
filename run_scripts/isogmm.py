import argparse
import os
import numpy as np
import glob
from cprint import *
import copy

from nigh_py import R3KDTree, SO3KDTree

from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import *


class ISOGMM:
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
        K_d = np.eye(3)
        K_d[0, 0] = 525.0/deci
        K_d[1, 1] = 525.0/deci
        K_d[0, 2] = 319.5/deci
        K_d[1, 2] = 239.5/deci
        self.W_d = (int)(640/deci)
        self.H_d = (int)(480/deci)
        self.iu_d = ImageUtils(K_d, im_h=self.H_d, im_w=self.W_d)

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

        # initialize KDTrees
        r3_radius = 2.0
        so3_radius = np.pi / 4.0
        self.r3_tree = R3KDTree(r3_radius)
        self.so3_tree = SO3KDTree(so3_radius)

        # other parameters
        self.l_thres = -100
        self.novel_pts_placeholder = None

        # results
        self.novel_pcld_o3d = None

    def novel_base(self, pcld, ids):
        scores = None
        for ci in ids:
            temp = self.sg.local_models[ci].score_samples(pcld)
            if scores is None:
                scores = temp
            else:
                scores = np.maximum(scores, temp)

        scores = scores.flatten()
        return pcld[scores < self.l_thres, :]

    def novel(self, pcld, kdtree_out):
        ids = [o[0].name for o in kdtree_out]
        return self.novel_base(pcld, ids)

    def process_frame(self, n):
        cprint.ok(f'frame {n}')

        if self.sg.model is not None:
            cprint.info(
                f'number of components in the global GMM model: {self.sg.model.n_components_}')

        # sensor information
        pose = self.traj[n].pose
        t = pose[:3, 3]
        R = pose[:3, :3]

        # point cloud information
        pcld, _ = self.iu_d.generate_pcld_wf(pose, rgb_path=self.rgb_paths[n],
                                             depth_path=self.depth_paths[n],
                                             size=(self.W_d, self.H_d))

        # overall novel points for this frame
        this_novel = None

        if self.sg.model is None:
            # no GMM model so far, all points are novel
            this_novel = pcld
        else:
            # find the closest GMM fragment
            r3_out = self.r3_tree.search(t)
            so3_out = self.so3_tree.search(R)

            if len(r3_out) == 0:
                cprint.warn(f'R3KDTree: no pose close to {t} frame {n}')
                cprint.info('Learning a new GMM for this frame.')
                this_novel = pcld
            elif len(so3_out) == 0:
                cprint.warn(
                    f'SO3KDTree could not find any pose close to {R} frame {n}')
                cprint.info('Using the output from R3KDTree')
                # get the novel points
                this_novel = self.novel(pcld, r3_out)
            else:
                # find the common index in the two kdtree outputs
                r3_ids = np.array([int(o[0].name) for o in r3_out])
                so3_ids = np.array([int(o[0].name) for o in so3_out])
                common_ids = np.intersect1d(r3_ids, so3_ids)

                if len(common_ids) == 0:
                    cprint.warn(
                        f'No common ids b/w R3 {r3_ids} and SO3 {so3_ids}')
                    cprint.info('Using the output from R3KDTree')
                    # get the novel points
                    this_novel = self.novel(pcld, r3_out)
                else:
                    cprint.info(f'Common ids b/w R3 and SO3: {common_ids}')
                    this_novel = self.novel_base(pcld, common_ids)

        # process novel points
        if self.novel_pts_placeholder is None:
            self.novel_pts_placeholder = copy.deepcopy(this_novel)
        else:
            self.novel_pts_placeholder = np.concatenate(
                (self.novel_pts_placeholder, this_novel), axis=0)

        self.novel_pcld_o3d = np_to_o3d_tensor(copy.deepcopy(self.novel_pts_placeholder[:, :3]),
                                               color=[1.0, 0.0, 0.0])

        # if the number is above a threshold, then only learn
        if self.novel_pts_placeholder.shape[0] >= (int)(self.W_d * self.H_d):
            cprint.info('Fitting Model.')
            # fit
            self.sg.fit(self.novel_pts_placeholder)
            # add to the KDTrees
            self.r3_tree.insert(t)
            self.so3_tree.insert(R.flatten())
            # reset
            self.novel_pts_placeholder = None
            # return True to indicate that the model was updated
            return True
        else:
            # go to next frame
            cprint.warn(
                'Number of points are not enough, moving to next frame.')
            # return False to indicate that the model was not updated
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Incremental-SOGMM")
    parser.add_argument('--path_datasets', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--nframes', type=int)
    parser.add_argument('--bandwidth', type=float)

    args = parser.parse_args()

    isogmm = ISOGMM(args.path_datasets,
                    args.dataset_name,
                    nframes=args.nframes,
                    bandwidth=args.bandwidth)

    isogmm.process_frame(0)
    print(isogmm.novel_pcld_o3d)
    isogmm.process_frame(1)
    print(isogmm.novel_pcld_o3d)
    isogmm.process_frame(2)
    print(isogmm.novel_pcld_o3d)
