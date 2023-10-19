import argparse
import os
import numpy as np
import glob
import time
from cprint import cprint

from sogmm_py.vis_open3d import VisOpen3D
from sogmm_py.utils import read_log_trajectory, np_to_o3d, o3d_to_np, ImageUtils
from sogmm_py.gmm_spatial_hash import GMMSpatialHash

from sogmm_gpu import SOGMMInference as GPUInference
from sogmm_gpu import SOGMMLearner as GPUFit
from sogmm_gpu import SOGMMf4Device as GPUContainerf4
from sogmm_cpu import SOGMMf4Host as CPUContainerf4


def extract_ms_data(X):
    d = np.array([np.linalg.norm(x) for x in X[:, 0:3]])[:, np.newaxis]
    g = X[:, 3][:, np.newaxis]
    return np.concatenate((d, g), axis=1)


parser = argparse.ArgumentParser(description="Incremental-SOGMM")
parser.add_argument('--datasetroot', type=str)

args = parser.parse_args()

dfolder = args.datasetroot

dname = 'lounge'
deci = 2.0
bandwidth = 0.015

learner = GPUFit(bandwidth)
inference = GPUInference()
gsh = GMMSpatialHash(resolution=0.2)

frame_1 = 675
color_1 = [1.0, 0.0, 0.0]
frame_2 = 700
color_2 = [0.0, 0.0, 1.0]

# paths to all rgb and depth images in the dataset
rgb_paths = sorted(
    glob.glob(os.path.join(dfolder, dname + '-color/*.png')))
depth_paths = sorted(
    glob.glob(os.path.join(dfolder, dname + '-depth/*.png')))

# read the full camera trajectory
traj = read_log_trajectory(os.path.join(dfolder, dname + '-traj.log'))

K_d = np.eye(3)
K_d[0, 0] = 525.0/deci
K_d[1, 1] = 525.0/deci
K_d[0, 2] = 319.5/deci
K_d[1, 2] = 239.5/deci
W_d = (int)(640/deci)
H_d = (int)(480/deci)
iu_d = ImageUtils(K_d, im_h=H_d, im_w=W_d)

# first frame
pose_1 = traj[frame_1].pose
translation_1 = pose_1[:3, 3]
rotation_1 = pose_1[:3, :3].flatten()
pcld_1, im_1 = iu_d.generate_pcld_wf(pose_1, rgb_path=rgb_paths[frame_1],
                                     depth_path=depth_paths[frame_1], size=(W_d, H_d))
cprint.info(f"num points pcld_1: {pcld_1.shape[0]}")

lsogmm_gpu = GPUContainerf4()
learner.fit(extract_ms_data(pcld_1), pcld_1, lsogmm_gpu)
cprint.info(f"num components lsogmm_gpu: {lsogmm_gpu.n_components_}")

# second frame
pose_2 = traj[frame_2].pose
translation_2 = pose_2[:3, 3]
pcld_2, im_2 = iu_d.generate_pcld_wf(pose_2, rgb_path=rgb_paths[frame_2],
                                     depth_path=depth_paths[frame_2], size=(W_d, H_d))
cprint.info(f"num points pcld_2: {pcld_2.shape[0]}")

ts = time.time()
scores_4d = inference.score_4d(pcld_2, lsogmm_gpu)
te = time.time()
cprint.info(f"4D time {te - ts} seconds")
scores_4d = scores_4d.flatten()
novel_4d_case = pcld_2[scores_4d < -3.14, :]

ts = time.time()
scores_3d = inference.score_3d(pcld_2[:, :3], lsogmm_gpu)
te = time.time()
cprint.info(f"3D time {te - ts} seconds")
scores_3d = scores_3d.flatten()
novel_3d_case = pcld_2[scores_3d < -3.14, :]

lsogmm_cpu = CPUContainerf4(lsogmm_gpu.n_components_)
lsogmm_gpu.to_host(lsogmm_cpu)
gsh.add_points(lsogmm_cpu.means_, np.arange(0, lsogmm_cpu.n_components_, dtype=int))

ts = time.time()
fov_comp_indices = gsh.find_points(pcld_2)
submap_cpu = lsogmm_cpu.submap_from_indices(fov_comp_indices)
submap_gpu = GPUContainerf4(submap_cpu.n_components_)
submap_gpu.from_host(submap_cpu)
scores_3d_fov = inference.score_3d(pcld_2[:, :3], submap_gpu)
te = time.time()
cprint.info(f"3D FoV time {te - ts} seconds")
cprint.info(f"num components submap_gpu: {submap_gpu.n_components_}")
scores_3d_fov = scores_3d_fov.flatten()
novel_3d_fov_case = pcld_2[scores_3d_fov < -3.14, :]

vis0 = VisOpen3D(visible=True, window_name='a')
vis0.add_geometries([np_to_o3d(pcld_1),
                    np_to_o3d(pcld_2)])

vis0.add_geometries([np_to_o3d(pcld_1[:, :3], color=color_1),
                    np_to_o3d(pcld_2[:, :3], color=color_2)])
vis0.frustrum(pose_1, K_d, W_d, H_d, scale=0.4, color=color_1)
vis0.frustrum(pose_2, K_d, W_d, H_d, scale=0.4, color=color_2)

vis1 = VisOpen3D(visible=True, window_name='b')
vis1.add_geometries([np_to_o3d(pcld_1),
                    np_to_o3d(novel_4d_case[:, :3], color=color_2)])
vis1.update_view_point(extrinsic=np.linalg.inv(pose_2))
vis1.poll_events()
vis1.update_renderer()

vis2 = VisOpen3D(visible=True, window_name='c')
vis2.add_geometries([np_to_o3d(pcld_1),
                    np_to_o3d(novel_3d_case[:, :3], color=color_2)])
vis2.update_view_point(extrinsic=np.linalg.inv(pose_2))
vis2.poll_events()
vis2.update_renderer()

vis3 = VisOpen3D(visible=True, window_name='d')
vis3.add_geometries([np_to_o3d(pcld_1),
                    np_to_o3d(novel_3d_fov_case[:, :3], color=color_2)])
vis3.update_view_point(extrinsic=np.linalg.inv(pose_2))
vis3.poll_events()
vis3.update_renderer()

vis0.run()
vis1.run()
vis2.run()
vis3.run()
