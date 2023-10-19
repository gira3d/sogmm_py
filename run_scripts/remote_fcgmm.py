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

from sogmm_gpu import SOGMMInference as GPUInference
from sogmm_cpu import SOGMMInference as CPUInference
from sogmm_gpu import SOGMMLearner as GPUFit
from sogmm_gpu import SOGMMf4Device as GPUContainerf4
from sogmm_cpu import SOGMMf4Host as CPUContainerf4


class FCGMManager:
    def __init__(self, n_components, connection, path_datasets, dataset_name, path_results, machine, zfill=5,
                 l_thres=3.0, color_ext="png", deci=5.0, inf_type='gpu'):
        self.path_datasets = path_datasets
        self.path_results = path_results
        self.dataset_name = dataset_name
        self.c_ext = color_ext
        self.machine = machine
        self.zfill = zfill

        self.inf_type = inf_type

        self.K = n_components

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

        self.learner = GPUFit()
        if self.inf_type == 'gpu':
            self.inference = GPUInference()
        else:
            self.inference = CPUInference()

        self.global_model = None

        # other parameters
        self.l_thres = l_thres
        self.novel_pts_placeholder = None

    def extract_novel(self, model, pcld):
        # perform likelihood score computation on the GPU
        try:
            scores = self.inference.score_4d(pcld, model)
        except RuntimeError:
            return None

        # filter the point cloud and return novel points
        scores = scores.flatten()
        return pcld[scores < self.l_thres, :]

    def process_frame(self, n):
        result = False

        cprint.ok(f'frame {n}')

        if self.global_model is not None:
            cprint.info(
                f'number of components in the global GMM model: {self.global_model.n_components_}')

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
        if self.global_model is None:
            this_model_gpu = GPUContainerf4()
            self.learner.fit_em(pcld, self.K, this_model_gpu)

            if self.inf_type == 'gpu':
                self.global_model = copy.deepcopy(this_model_gpu)
            else:
                this_model_cpu = CPUContainerf4(this_model_gpu.n_components_)
                this_model_gpu.to_host(this_model_cpu)
                self.global_model = copy.deepcopy(this_model_cpu)

            # a new gmm was learnt
            result = True
        else:
            ts = time.time()

            novel_pts = self.extract_novel(self.global_model,
                                           pcld)
            if novel_pts is None:
                return False, None

            te = time.time()
            ttaken = te - ts

            # process novel points
            if self.novel_pts_placeholder is None:
                self.novel_pts_placeholder = copy.deepcopy(novel_pts)
            else:
                self.novel_pts_placeholder = np.concatenate(
                    (self.novel_pts_placeholder, novel_pts), axis=0)

            if self.novel_pts_placeholder.shape[0] >= (int)((640 / self.deci) * (480 / self.deci)):
                this_model_gpu = GPUContainerf4()
                self.learner.fit_em(
                    self.novel_pts_placeholder, self.K, this_model_gpu)

                if self.inf_type == 'gpu':
                    self.global_model.merge(this_model_gpu)
                else:
                    this_model_cpu = CPUContainerf4(this_model_gpu.n_components_)
                    this_model_gpu.to_host(this_model_cpu)
                    self.global_model.merge(this_model_cpu)

                self.novel_pts_placeholder = None

                # a new gmm was learnt
                result = True
            else:
                # no new gmm was learnt
                result = False

        return result, ttaken


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--components', type=int, default=100)
    parser.add_argument('--inf_type', type=str, default="gpu")

    args = parser.get_config()

    c = Connection(host='192.168.3.182', user='fractal')
    fcm = FCGMManager(args.components, c, args.path_datasets,
                      args.dataset_name,
                      args.path_results,
                      args.machine,
                      args.zfill,
                      args.l_thres,
                      color_ext=args.color_ext,
                      deci=args.deci,
                      inf_type=args.inf_type)

    max_frame = 0
    gmm_learnt = []
    time_takens = []
    for i in range(0, args.nframes):
        cprint.info(f"gmm in frame {i}")
        result, tt = fcm.process_frame(i)
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

    model_file = f"{fcm.dataset_name}_model_{fcm.K}_{args.inf_type}.pkl"
    if args.inf_type == 'gpu':
        cpu_model = CPUContainerf4(fcm.global_model.n_components_)
        fcm.global_model.to_host(cpu_model)
        with open(model_file, 'wb') as f:
            pickle.dump(cpu_model, f)
    else:
        with open(model_file, 'wb') as f:
            pickle.dump(fcm.global_model, f)

    fcm.connection.put(model_file, remote=os.path.join(
        fcm.path_results, f"fcgmm_results/"))
    os.remove(model_file)

    metrics_file = f"{fcm.dataset_name}_{fcm.K}_{args.inf_type}.csv"
    df = pd.DataFrame(metrics, columns=['frame', 'time', 'gmmlearnt'])
    df.set_index('frame')
    df.to_csv(metrics_file)
    fcm.connection.put(metrics_file, remote=os.path.join(
        fcm.path_results, f"fcgmm_results/"))
    os.remove(metrics_file)
