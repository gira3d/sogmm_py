import argparse
import os
import numpy as np
import glob
from cprint import *
import copy

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import *

import sogmm_cpu
from sogmm_cpu import SOGMMf4Host as CPUContainerf4
from sogmm_cpu import SOGMMf3Host as CPUContainerf3

import sogmm_gpu
from sogmm_gpu import SOGMMf4Device as GPUContainerf4
from sogmm_gpu import SOGMMf3Device as GPUContainerf3
from sogmm_gpu import SOGMMLearner as GPUFit
from sogmm_gpu import SOGMMInference as GPUInference


class GMMSpatialHash:
    def __init__(self, width=100, height=100, depth=100, resolution=0.2):
        self.w = (int)(width)
        self.h = (int)(height)
        self.d = (int)(depth)
        self.res = resolution
        self.o = (-1.0 * resolution / 2.0) * np.array([width, height, depth])

        self.hash_table = {}

    def add_point(self, key, value):
        idx = self.point_to_index(key)

        try:
            temp = self.hash_table[idx]
        except KeyError:
            self.hash_table[idx] = list()

        self.hash_table[idx].append(value)

    def add_points(self, points, values):
        N = points.shape[0]
        for i in range(N):
            self.add_point(points[i, :3], values[i])

    def point_to_index(self, point):
        r = (int)((point[1] - self.o[1]) / self.res)
        c = (int)((point[0] - self.o[0]) / self.res)
        s = (int)((point[2] - self.o[2]) / self.res)

        return (r*self.w + c)*self.d + s

    def find_point(self, key):
        idx = self.point_to_index(key)

        try:
            temp = self.hash_table[idx]
            return temp
        except KeyError:
            return [-1]

    def find_points(self, points):
        N = points.shape[0]
        results = []
        for i in range(N):
            results += self.find_point(points[i, :3])

        results = np.array(results, dtype=int)

        return np.unique(results[results >= 0])


class GMMSpatialHashManager:
    def __init__(self, path_datasets, dataset_name, nframes=None, bandwidth=0.025, deci=8.0, compute='GPU'):
        # paths to all rgb and depth images in the dataset
        self.rgb_paths = sorted(
            glob.glob(os.path.join(path_datasets, dataset_name + '-color/*.jpg')))
        self.depth_paths = sorted(
            glob.glob(os.path.join(path_datasets, dataset_name + '-depth/*.png')))

        self.max_frames = len(self.rgb_paths)

        # read the full camera trajectory
        self.traj = read_log_trajectory(os.path.join(
            path_datasets, dataset_name + '-traj.log'))

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

        self.window = gui.Application.instance.create_window(
            'GIRA3D - Reconstruction', 1920, 1080)

        self.widget3d = gui.SceneWidget()

        self.window.add_child(self.widget3d)
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([0, 0, 0, 1])

        self.window.set_needs_layout()
        self.widget3d.look_at([0, 0, 0], [0, 1, 3], [0, 0, -1])

        # hash table
        self.gsh = GMMSpatialHash()

        # other parameters
        self.l_thres = 3.0
        self.novel_pts_placeholder = None

    def extract_novel(self, model, pcld, comp_indices):
        # create a GMM submap using component indices
        submap = model.submap_from_indices(list(comp_indices))

        # take it to the GPU
        submap_gpu = GPUContainerf4(submap.n_components_)
        submap_gpu.from_host(submap)

        # perform likelihood score computation on the GPU
        scores = self.inference.score_3d(pcld[:, :3], submap_gpu)

        # filter the point cloud and return novel points
        scores = scores.flatten()
        return pcld[scores < self.l_thres, :]

    def extract_ms_data(self, X):
        d = np.array([np.linalg.norm(x) for x in X[:, 0:3]])[:, np.newaxis]
        g = X[:, 3][:, np.newaxis]
        return np.concatenate((d, g), axis=1)

    def process_frame(self, n):
        cprint.ok(f'frame {n}')

        if self.cpu_model is not None:
            cprint.info(
                f'number of components in the global GMM model: {self.cpu_model.n_components_}')

        # sensor information
        pose = self.traj[n].pose

        # point cloud information
        pcld, _ = self.iu_d.generate_pcld_wf(pose, rgb_path=self.rgb_paths[n],
                                             depth_path=self.depth_paths[n],
                                             size=(self.W_d, self.H_d))
        if self.cpu_model is None:
            this_model_gpu = GPUContainerf4()
            self.learner.fit(self.extract_ms_data(pcld), pcld, this_model_gpu)

            this_model_cpu = CPUContainerf4(this_model_gpu.n_components_)
            this_model_gpu.to_host(this_model_cpu)

            self.gsh.add_points(this_model_cpu.means_, np.arange(
                0, this_model_cpu.n_components_, dtype=int))

            self.cpu_model = copy.deepcopy(this_model_cpu)

            self.visualize_frustum(n, pose, color=[1.0, 0.0, 0.0])
            self.visualize_resampled_pcld(n, this_model_gpu)
        else:
            fov_comp_indices = self.gsh.find_points(pcld)

            novel_pts = None
            if len(fov_comp_indices) > 1:
                novel_pts = self.extract_novel(self.cpu_model,
                                               pcld,
                                               fov_comp_indices)
            else:
                novel_pts = pcld

            # process novel points
            if self.novel_pts_placeholder is None:
                self.novel_pts_placeholder = copy.deepcopy(novel_pts)
            else:
                self.novel_pts_placeholder = np.concatenate(
                    (self.novel_pts_placeholder, novel_pts), axis=0)

            if self.novel_pts_placeholder.shape[0] >= (int)((640 / self.deci) * (480 / self.deci)):
                old_n_components = self.cpu_model.n_components_

                this_model_gpu = GPUContainerf4()
                self.learner.fit(self.extract_ms_data(self.novel_pts_placeholder),
                                 self.novel_pts_placeholder, this_model_gpu)
                this_model_cpu = CPUContainerf4(this_model_gpu.n_components_)
                this_model_gpu.to_host(this_model_cpu)

                self.cpu_model.merge(this_model_cpu)

                new_n_components = self.cpu_model.n_components_

                self.gsh.add_points(this_model_cpu.means_,
                                    np.arange(old_n_components,
                                              new_n_components,
                                              dtype=int))

                self.visualize_resampled_pcld(n, this_model_gpu)
                self.visualize_frustum(n, pose, color=[1.0, 0.0, 0.0])

                self.novel_pts_placeholder = None

    def visualize_resampled_pcld(self, n, model):
        resampled = np_to_o3d_tensor(
            self.inference.reconstruct(model, 640*480, 2.2))
        mat = rendering.MaterialRecord()
        # mat.shader = 'defaultUnlit'
        # mat.sRGB_color = True
        self.widget3d.scene.scene.add_geometry('resampled_' + str(n),
                                               resampled,
                                               mat)

    def visualize_frustum(self, n, pose, color=[0.961, 0.475, 0.000]):
        frustum = o3d.geometry.LineSet.create_camera_visualization(
            self.W_d, self.H_d, self.K_d,
            np.linalg.inv(pose), 0.2)
        frustum.paint_uniform_color(color)
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 5.0
        self.widget3d.scene.add_geometry("frustum_" + str(n), frustum, mat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Incremental-SOGMM")
    parser.add_argument('--path_datasets', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--nframes', type=int)
    parser.add_argument('--bandwidth', type=float)
    parser.add_argument('--deci', type=float)

    args = parser.parse_args()

    app = gui.Application.instance
    app.initialize()

    gshm = GMMSpatialHashManager(args.path_datasets,
                                 args.dataset_name,
                                 bandwidth=args.bandwidth,
                                 deci=args.deci)

    if args.nframes > gshm.max_frames:
        args.nframes = gshm.max_frames

    for i in range(0, args.nframes):
        cprint.info(f"gmm in frame {i}")
        gshm.process_frame(i)

    app.run()
