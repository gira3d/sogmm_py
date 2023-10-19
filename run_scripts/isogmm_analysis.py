import os
import pickle
from cprint import cprint
import pandas as pd
import time
import numpy as np

import open3d as o3d
from fabric import Connection

from sogmm_py.utils import *
from sogmm_py.vis_open3d import VisOpen3D
from config_parser import ConfigParser
from sogmm_py.gmm_spatial_hash import GMMSpatialHash

import sogmm_cpu
from sogmm_cpu import SOGMMInference as CPUInference


def comp_validity_check(cov):
    # if two smallest semi-axis lengths are smaller than a threshold, ignore the component
    eigvals, _ = np.linalg.eigh(cov)
    sorted_eigvals = np.sort(eigvals)

    if np.sqrt(sorted_eigvals[0]) < 1e-4 and np.sqrt(sorted_eigvals[1]) < 1e-4:
        return False
    else:
        return True


class ISOGMMAnalysis:
    def __init__(self, connection, path_datasets, dataset_name,
                 path_results, machine, bw_list,
                 zfill=5, nframes=2870, color_ext="png"):
        self.path_datasets = path_datasets
        self.dataset_name = dataset_name
        self.path_results = path_results
        self.machine = machine
        self.c_ext = color_ext
        self.zfill = zfill
        self.nframes = nframes

        self.connection = connection

        traj_path = os.path.join(
            self.path_datasets, f"{dataset_name}-traj.log")
        self.connection.get(traj_path)
        self.traj = read_log_trajectory(f"{dataset_name}-traj.log")
        os.remove(f"{dataset_name}-traj.log")

        self.machine_results = os.path.join(self.path_results, self.machine)
        self.bw_list = bw_list
        self.bw_str = lambda x: f"bw_{str(x).replace('.', '_')}"
        self.csv_name = lambda a: os.path.join(self.machine_results,
                                               f"{self.dataset_name}_{self.bw_str(a)}.csv")
        self.model_name = lambda a: os.path.join(self.machine_results,
                                                 f"{self.dataset_name}_model_{self.bw_str(a)}.pkl")

        self.model_files = [self.model_name(
            a) for a in self.bw_list if os.path.isfile(self.model_name(a))]

        d = 4
        self.K = np.eye(3)
        self.K[0, 0] = 525.0/d
        self.K[1, 1] = 525.0/d
        self.K[0, 2] = 319.5/d
        self.K[1, 2] = 239.5/d
        self.W = (int)(640/d)
        self.H = (int)(480/d)
        self.iu = ImageUtils(self.K, im_h=self.H, im_w=self.W)

        self.frame_str = lambda x: str(x).zfill(self.zfill)
        self.rgb_path_getter = lambda n: os.path.join(self.path_datasets,
                                                      f"{self.dataset_name}-color",
                                                      f"{self.frame_str(n)}.{self.c_ext}")
        self.depth_path_getter = lambda n: os.path.join(self.path_datasets,
                                                        f"{self.dataset_name}-depth",
                                                        f"{self.frame_str(n)}.png")

        self.infer = CPUInference()

    def create_gt(self, v):
        gt_pcds = []
        gt_Ns = []
        for n in range(0, self.nframes):
            # get pose for this frame
            pose = self.traj[n].pose
            # get the gt image and pcld
            rgb_file = self.rgb_path_getter(n)
            c.get(rgb_file, local=f"{self.frame_str(n)}-color.{self.c_ext}")
            depth_file = self.depth_path_getter(n)
            c.get(depth_file, local=f"{self.frame_str(n)}-depth.png")
            gt_pcld, _ = self.iu.generate_pcld_wf(pose, rgb_path=f"{self.frame_str(n)}-color.{self.c_ext}",
                                                  depth_path=f"{self.frame_str(n)}-depth.png",
                                                  size=(self.W, self.H))
            os.remove(f"{self.frame_str(n)}-color.{self.c_ext}")
            os.remove(f"{self.frame_str(n)}-depth.png")

            gt_pcds.append(gt_pcld)
            gt_Ns.append(gt_pcld.shape[0])

            cprint.ok(f"calc. gt pcld {n}")

        full_gt_pcld = np.zeros((sum(gt_Ns), 4))
        prev_i = 0
        for i in range(0, len(gt_pcds)):
            n = gt_Ns[i]
            assert (n == gt_pcds[i].shape[0])
            full_gt_pcld[prev_i:prev_i+n, :] = gt_pcds[i]
            prev_i += n

        gt_pcld_downsampled = np_to_o3d(
            full_gt_pcld).voxel_down_sample(voxel_size=v)
        o3d.io.write_point_cloud(
            f"{self.dataset_name}_gt_voxelized.pcd", gt_pcld_downsampled)

    def reconstruct(self, v, s):
        gt_pcld_downsampled = o3d.io.read_point_cloud(
            f"{self.dataset_name}_gt_voxelized.pcd")

        num_results = len(self.bw_list)
        metrics = np.zeros((num_results, 6))
        for i, a in enumerate(self.bw_list):
            mf = self.model_name(a)
            self.connection.get(mf)
            sogmm = None
            with open(f"{self.dataset_name}_model_{self.bw_str(a)}.pkl", 'rb') as f:
                sogmm = pickle.load(f)
            os.remove(f"{self.dataset_name}_model_{self.bw_str(a)}.pkl")

            batch_size = 100
            comp_list = list(np.arange(sogmm.n_components_))

            ts = time.time()
            sogmm3 = sogmm_cpu.marginal_X(sogmm)
            covariances = matrix_to_tensor(sogmm3.covariances_, 3)
            pr_pcds = []
            pr_Ns = []
            for k in range(0, sogmm.n_components_, batch_size):
                sub_comps = comp_list[k:k+batch_size]
                valid = [comp_validity_check(covariances[a])
                         for a in sub_comps]
                submap = sogmm.submap_from_indices(
                    [sub_comps[a] for a in range(len(sub_comps)) if valid[a] is True])

                pcld = self.infer.reconstruct_fast(
                    submap, submap.support_size_, s)
                pr_pcds.append(pcld)
                pr_Ns.append(pcld.shape[0])
            te = time.time()
            cprint.ok(f"time taken {te - ts} seconds.")

            full_pr_pcld = np.zeros((sum(pr_Ns), 4))
            prev_i = 0
            for j in range(0, len(pr_pcds)):
                n = pr_Ns[j]
                assert (n == pr_pcds[j].shape[0])
                full_pr_pcld[prev_i:prev_i+n, :] = pr_pcds[j]
                prev_i += n

            pr_pcld_downsampled = np_to_o3d(
                full_pr_pcld).voxel_down_sample(voxel_size=v)

            pr_pcld_file = f"{self.dataset_name}_{self.bw_str(a)}_pr_voxelized.pcd"
            o3d.io.write_point_cloud(pr_pcld_file, pr_pcld_downsampled)
            self.connection.put(pr_pcld_file,
                                remote=os.path.join(args.path_results, f"{args.machine}/"))
            os.remove(pr_pcld_file)

            result = calculate_all_metrics(
                gt_pcld_downsampled, pr_pcld_downsampled, th=v)

            metrics[i, 0] = self.bw_list[i]
            metrics[i, 1] = result[0]  # f-score
            metrics[i, 2] = result[1]  # precision
            metrics[i, 3] = result[2]  # recall
            metrics[i, 4] = result[3]  # recon. err. mean
            metrics[i, 5] = result[4]  # psnr

        return metrics


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--voxel_size', type=float, default=0.01)
    parser.add_argument('--sigma', type=float, default=1.8)

    args = parser.get_config()

    c = Connection(host='192.168.3.182', user='fractal')

    ianalysis = ISOGMMAnalysis(c, args.path_datasets,
                               args.dataset_name,
                               args.path_results,
                               args.machine,
                               args.bw_list,
                               args.zfill,
                               nframes=args.nframes,
                               color_ext=args.color_ext)

    # ianalysis.create_gt(v=0.01)
    metrics = ianalysis.reconstruct(v=args.voxel_size, s=args.sigma)

    for l in range(len(ianalysis.bw_list)):
        cprint.info(f"bw {metrics[l, 0]}\n"
                    "--------------------------\n"
                    f"f-score {(metrics[l, 1]):.2f}\n"
                    f"precision {(metrics[l, 2]):.2f}\n"
                    f"recall {(metrics[l, 3]):.2f}\n"
                    f"mre {(metrics[l, 4]):.4f}\n"
                    f"psnr {(metrics[l, 5]):.2f}\n")

    df = pd.DataFrame(metrics, columns=['bw',
                                        'f-score',
                                        'precision',
                                        'recall',
                                        'mre',
                                        'psnr'])
    df.set_index('bw')
    metrics_file = f"{args.dataset_name}_recon_metrics.csv"
    df.to_csv(metrics_file)
    c.put(metrics_file, remote=os.path.join(
        args.path_results, f"{args.machine}/"))
    os.remove(metrics_file)
