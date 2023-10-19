import os
import copy
import pickle
import time
import pandas as pd

from cprint import *
import numpy as np
from fabric import Connection

from sogmm_py.utils import *
from config_parser import ConfigParser
from sogmm_py.gmm_spatial_hash import GMMSpatialHash

import sogmm_cpu
from sogmm_gpu import SOGMMInference as GPUInference
from sogmm_gpu import SOGMMLearner as GPUFit
from sogmm_gpu import SOGMMf4Device as GPUContainerf4
from sogmm_cpu import SOGMMf4Host as CPUContainerf4


class GMMSpatialHashManager:
    def __init__(self, connection, path_datasets, dataset_name, path_results, machine, alpha=0.2, zfill=5,
                 l_thres=3.0, color_ext="png", bandwidth=0.025, deci=5.0):
        self.path_datasets = path_datasets
        self.path_results = path_results
        self.dataset_name = dataset_name
        self.c_ext = color_ext
        self.machine = machine
        self.zfill = zfill

        self.connection = connection

        # get the trajectory file
        self.connection.get(os.path.join(
            self.path_datasets, f"{self.dataset_name}-traj.log"))

        # read the full camera trajectory
        self.traj = read_log_trajectory(f"{dataset_name}-traj.log")

        os.remove(f"{dataset_name}-traj.log")

        # image/pcld utility
        self.deci = deci
        self.K_d = np.eye(3)
        self.K_d[0, 0] = 525.0/deci
        self.K_d[1, 1] = 525.0/deci
        self.K_d[0, 2] = 319.5/deci
        self.K_d[1, 2] = 239.5/deci
        self.W_d = (int)(640/deci)
        self.H_d = (int)(480/deci)
        self.iu_d = ImageUtils(self.K_d, im_h=self.H_d, im_w=self.W_d)

        self.bandwidth = bandwidth
        self.learner = GPUFit(self.bandwidth)
        self.inference = GPUInference()

        self.cpu_model = None

        # hash table
        self.gsh = GMMSpatialHash(resolution=alpha)

        # other parameters
        self.l_thres = l_thres
        self.novel_pts_placeholder = None

    def extract_novel(self, model, pcld, comp_indices):
        # create a GMM submap using component indices
        submap = model.submap_from_indices(list(comp_indices))

        # take it to the GPU
        submap_gpu = GPUContainerf4(submap.n_components_)
        submap_gpu.from_host(submap)

        # perform likelihood score computation on the GPU
        try:
            scores = self.inference.score_3d(pcld[:, :3], submap_gpu)
        except RuntimeError:
            cprint.warn("Ran out of GPU RAM.")
            return None

        # filter the point cloud and return novel points
        scores = scores.flatten()
        return pcld[scores < self.l_thres, :]

    def extract_ms_data(self, X):
        d = np.array([np.linalg.norm(x) for x in X[:, 0:3]])[:, np.newaxis]
        g = X[:, 3][:, np.newaxis]
        return np.concatenate((d, g), axis=1)

    def process_frame(self, n):
        result = False

        cprint.ok(f'frame {n}')

        if self.cpu_model is not None:
            cprint.info(
                f'number of components in the global GMM model: {self.cpu_model.n_components_}')

        # sensor information
        pose = self.traj[n].pose

        # point cloud information
        frame_str = str(n).zfill(self.zfill)
        rgb_im_path = f"{frame_str}-color.{self.c_ext}"
        depth_im_path = f"{frame_str}-depth.png"
        self.connection.get(os.path.join(self.path_datasets,
                                         f"{self.dataset_name}-color", f"{frame_str}.{self.c_ext}"),
                            local=rgb_im_path)
        self.connection.get(os.path.join(self.path_datasets,
                                         f"{self.dataset_name}-depth", f"{frame_str}.png"),
                            local=depth_im_path)

        pcld, _ = self.iu_d.generate_pcld_wf(pose, rgb_path=rgb_im_path,
                                             depth_path=depth_im_path,
                                             size=(self.W_d, self.H_d))

        os.remove(rgb_im_path)
        os.remove(depth_im_path)

        ttaken = None
        if self.cpu_model is None:
            this_model_gpu = GPUContainerf4()
            try:
                self.learner.fit(self.extract_ms_data(pcld), pcld, this_model_gpu)
            except RuntimeError:
                cprint.warn('Ran out of GPU RAM.')
                return False, None

            this_model_cpu = CPUContainerf4(this_model_gpu.n_components_)
            this_model_gpu.to_host(this_model_cpu)

            this_model_cpu3 = sogmm_cpu.marginal_X(this_model_cpu)
            self.gsh.add_sigma_points(this_model_cpu3.means_, matrix_to_tensor(this_model_cpu3.covariances_, 3),
                                      np.arange(0, this_model_cpu.n_components_, dtype=int))

            self.cpu_model = copy.deepcopy(this_model_cpu)

            # a new gmm was learnt
            result = True
        else:
            ts = time.time()
            fov_comp_indices = self.gsh.find_points(pcld)

            novel_pts = None
            if len(fov_comp_indices) > 1:
                novel_pts = self.extract_novel(self.cpu_model,
                                               pcld,
                                               fov_comp_indices)
            else:
                novel_pts = pcld

            te = time.time()
            ttaken = te - ts

            if novel_pts is None:
                return False, ttaken

            # process novel points
            if self.novel_pts_placeholder is None:
                self.novel_pts_placeholder = copy.deepcopy(novel_pts)
            else:
                self.novel_pts_placeholder = np.concatenate(
                    (self.novel_pts_placeholder, novel_pts), axis=0)

            if self.novel_pts_placeholder.shape[0] >= (int)((640 / self.deci) * (480 / self.deci)):
                old_n_components = self.cpu_model.n_components_

                this_model_gpu = GPUContainerf4()
                try:
                    self.learner.fit(self.extract_ms_data(self.novel_pts_placeholder),
                                    self.novel_pts_placeholder, this_model_gpu)
                except RuntimeError:
                    cprint.warn('Ran out of GPU RAM.')
                    return False, None
                this_model_cpu = CPUContainerf4(this_model_gpu.n_components_)
                this_model_gpu.to_host(this_model_cpu)

                self.cpu_model.merge(this_model_cpu)

                new_n_components = self.cpu_model.n_components_

                this_model_cpu3 = sogmm_cpu.marginal_X(this_model_cpu)
                self.gsh.add_sigma_points(this_model_cpu3.means_, matrix_to_tensor(this_model_cpu3.covariances_, 3),
                                        np.arange(old_n_components, new_n_components, dtype=int))

                self.novel_pts_placeholder = None

                # a new gmm was learnt
                result = True
            else:
                # continue to next frame or whatever
                result = False

        return result, ttaken


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--alpha', type=float, default=0.2)

    args = parser.get_config()

    c = Connection(host='192.168.3.182', user='fractal')
    gshm = GMMSpatialHashManager(c, args.path_datasets,
                                 args.dataset_name,
                                 args.path_results,
                                 args.machine,
                                 args.alpha,
                                 args.zfill,
                                 args.l_thres,
                                 color_ext=args.color_ext,
                                 bandwidth=args.bandwidth,
                                 deci=args.deci)

    max_frame = 0
    gmm_learnt = []
    time_takens = []
    for i in range(0, args.nframes):
        cprint.info(f"gmm in frame {i}")
        result, tt = gshm.process_frame(i)
        if tt is None:
            time_takens.append(-1.0)
        else:
            time_takens.append(tt)
        if result:
            # a new model was learnt
            gmm_learnt.append(1)
        else:
            # only the novelty check was performed
            gmm_learnt.append(0)

        max_frame = (int)(i)

        if result is False and tt is None:
            break

    # frame number, time taken, gmm learnt or not
    metrics = np.zeros((max_frame, 3))
    for j in range(0, max_frame):
        metrics[j, 0] = (int)(j)
        metrics[j, 1] = time_takens[j]
        metrics[j, 2] = gmm_learnt[j]

    bw_str = f"bw_{str(gshm.bandwidth).replace('.', '_')}_alpha_{str(args.alpha).replace('.', '_')}"

    model_file = f"{gshm.dataset_name}_model_{bw_str}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(gshm.cpu_model, f)
    gshm.connection.put(model_file, remote=os.path.join(
        gshm.path_results, f"isogmm_results_ll/"))
    os.remove(model_file)

    metrics_file = f"{gshm.dataset_name}_{bw_str}.csv"
    df = pd.DataFrame(metrics, columns=['frame', 'time', 'gmmlearnt'])
    df.set_index('frame')
    df.to_csv(metrics_file)
    gshm.connection.put(metrics_file, remote=os.path.join(
        gshm.path_results, f"isogmm_results_ll/"))
    os.remove(metrics_file)
