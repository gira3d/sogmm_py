import os
import numpy as np
import open3d as o3d
from termcolor import cprint
import pickle
from sogmm_py.vis_open3d import VisOpen3D
import csv
import matplotlib.pyplot as plt
import tikzplotlib as tz

from sogmm_py.utils import *

def read_from_csv(file):
    ret = {}
    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        fn = csv_reader.fieldnames
        for row in csv_reader:
            ret[float(row[fn[0]])] = float(row[fn[1]])
    return ret

# general params
tikz_save = True

# used in open3d visualization
sigma_sample = 1.8
sample_mult = 2
O3D_K = np.array([[935.30743609,   0.,         959.5],
                    [0.,         935.30743609, 539.5],
                    [0.,           0.,           1.]])

dataset_path = '/Volumes/T7/datasets'
results_path = '/Volumes/GoogleDrive/My Drive/phd/adaptive_perception/results'
paper_fig_path = '/Users/kshitijgoel/Documents/main/papers.nosync/ongoing/Paper-PRI-KG/data_analysis/figures/'

# qualitative analysis
# wall
vis = VisOpen3D(visible=False)
pcd_path = os.path.join(results_path, 'stonewall_sogmm/stonewall_0.01/local_pcd_380.pcd')
gt_pcd = o3d.io.read_point_cloud(pcd_path, format='pcd')
vis.add_geometry(gt_pcd)
traj = read_log_trajectory(os.path.join(dataset_path, 'stonewall-traj.log'))
gt_pose = traj[380].pose
vis.update_view_point(O3D_K, np.linalg.inv(gt_pose))
vis.update_renderer()
vis.capture_screen_image('wall.png')
del vis

model_path = os.path.join(results_path, 'stonewall_sogmm/stonewall_0.01/full_model.pkl')
with open(model_path, 'rb') as mf:
    model = pickle.load(mf)

recon_pcd = model.sample(sample_mult*model.support_size_, sigma_sample)
cprint('num components in model for stonewall %d' % (model.n_components_), 'green')
cprint('num points reconstructed %d' % (recon_pcd.shape[0]), 'green')

vis = VisOpen3D(visible=False)
vis.add_geometry(np_to_o3d(recon_pcd))
vis.update_view_point(O3D_K, np.linalg.inv(gt_pose))
vis.update_renderer()
vis.capture_screen_image('wall_recon.png')
del vis

# copier
vis = VisOpen3D(visible=False)
pcd_path = os.path.join(results_path, 'copyroom_sogmm/copyroom_0.01/local_pcd_9.pcd')
gt_pcd = o3d.io.read_point_cloud(pcd_path, format='pcd')
vis.add_geometry(gt_pcd)
traj = read_log_trajectory(os.path.join(dataset_path, 'copyroom-traj.log'))
gt_pose = traj[9].pose
vis.update_view_point(O3D_K, np.linalg.inv(gt_pose))
vis.update_renderer()
vis.capture_screen_image('copier.png')
del vis

model_path = os.path.join(results_path, 'copyroom_sogmm/copyroom_0.01/full_model.pkl')
with open(model_path, 'rb') as mf:
    model = pickle.load(mf)

recon_pcd = model.sample(sample_mult*model.support_size_, sigma_sample)
cprint('num components in model for copyroom %d' % (model.n_components_), 'green')
cprint('num points reconstructed %d' % (recon_pcd.shape[0]), 'green')

vis = VisOpen3D(visible=False)
vis.add_geometry(np_to_o3d(recon_pcd))
vis.update_view_point(O3D_K, np.linalg.inv(gt_pose))
vis.update_renderer()
vis.capture_screen_image('copier_recon.png')
del vis

# plant
vis = VisOpen3D(visible=False)
pcd_path = os.path.join(results_path, 'lounge_sogmm/lounge_0.01/local_pcd_640.pcd')
gt_pcd = o3d.io.read_point_cloud(pcd_path, format='pcd')
vis.add_geometry(gt_pcd)
traj = read_log_trajectory(os.path.join(dataset_path, 'lounge-traj.log'))
gt_pose = traj[640].pose
vis.update_view_point(O3D_K, np.linalg.inv(gt_pose))
vis.update_renderer()
vis.capture_screen_image('plant.png')
del vis

model_path = os.path.join(results_path, 'lounge_sogmm/lounge_0.01/full_model.pkl')
with open(model_path, 'rb') as mf:
    model = pickle.load(mf)

recon_pcd = model.sample(sample_mult*model.support_size_, sigma_sample)
cprint('num components in model for lounge %d' % (model.n_components_), 'green')
cprint('num points reconstructed %d' % (recon_pcd.shape[0]), 'green')

vis = VisOpen3D(visible=False)
vis.add_geometry(np_to_o3d(recon_pcd))
vis.update_view_point(O3D_K, np.linalg.inv(gt_pose))
vis.update_renderer()
vis.capture_screen_image('plant_recon.png')
del vis


# quantitative analysis
psnr_fig, psnr_ax = plt.subplots()
err_fig, err_ax = plt.subplots()
mem_fig, mem_ax = plt.subplots()
xticklabels = []
width = 0.2
colors = ['crimson', 'navy', 'darkcyan']

# OctoMap
psnr_path_wall = os.path.join(results_path, 'stonewall_octomap/psnr.csv')
psnr_path_copier = os.path.join(results_path, 'copyroom_octomap/psnr.csv')
psnr_path_plant = os.path.join(results_path, 'lounge_octomap/psnr.csv')

wall_psnr_om = read_from_csv(psnr_path_wall)
copier_psnr_om = read_from_csv(psnr_path_copier)
plant_psnr_om = read_from_csv(psnr_path_plant)

idx = np.asarray([i for i in range(0, 2)])
psnr_ax.bar(idx - width, [val for _, val in sorted(wall_psnr_om.items(), reverse=True)], width=width, color=colors[0])
psnr_ax.bar(idx, [val for _, val in sorted(copier_psnr_om.items(), reverse=True)], width=width, color=colors[1])
psnr_ax.bar(idx + width, [val for _, val in sorted(plant_psnr_om.items(), reverse=True)], width=width, color=colors[2])

err_path_wall = os.path.join(results_path, 'stonewall_octomap/recon_mean.csv')
err_path_copier = os.path.join(results_path, 'copyroom_octomap/recon_mean.csv')
err_path_plant = os.path.join(results_path, 'lounge_octomap/recon_mean.csv')

wall_err_om = read_from_csv(err_path_wall)
copier_err_om = read_from_csv(err_path_copier)
plant_err_om = read_from_csv(err_path_plant)

idx = np.asarray([i for i in range(0, 2)])
err_ax.bar(idx - width, [val for _, val in sorted(wall_err_om.items(), reverse=True)], width=width, color=colors[0])
err_ax.bar(idx, [val for _, val in sorted(copier_err_om.items(), reverse=True)], width=width, color=colors[1])
err_ax.bar(idx + width, [val for _, val in sorted(plant_err_om.items(), reverse=True)], width=width, color=colors[2])

mem_path_wall = os.path.join(results_path, 'stonewall_octomap/mem.csv')
mem_path_copier = os.path.join(results_path, 'copyroom_octomap/mem.csv')
mem_path_plant = os.path.join(results_path, 'lounge_octomap/mem.csv')

wall_mem_om = read_from_csv(mem_path_wall)
copier_mem_om = read_from_csv(mem_path_copier)
plant_mem_om = read_from_csv(mem_path_plant)

idx = np.asarray([i for i in range(0, 2)])
mem_ax.bar(idx - width, [val / 1e6 for _, val in sorted(wall_mem_om.items(), reverse=True)], width=width, color=colors[0])
mem_ax.bar(idx, [val / 1e6 for _, val in sorted(copier_mem_om.items(), reverse=True)], width=width, color=colors[1])
mem_ax.bar(idx + width, [val / 1e6 for _, val in sorted(plant_mem_om.items(), reverse=True)], width=width, color=colors[2])

xticklabels += ['OM ' + str(s) for s, _ in sorted(wall_psnr_om.items(), reverse=True)]

# NDTMap
psnr_path_wall = os.path.join(results_path, 'stonewall_ndtmap/psnr.csv')
psnr_path_copier = os.path.join(results_path, 'copyroom_ndtmap/psnr.csv')
psnr_path_plant = os.path.join(results_path, 'lounge_ndtmap/psnr.csv')

wall_psnr_ndt = read_from_csv(psnr_path_wall)
copier_psnr_ndt = read_from_csv(psnr_path_copier)
plant_psnr_ndt = read_from_csv(psnr_path_plant)

idx = np.asarray([i for i in range(2, 4)])
psnr_ax.bar(idx - width, [val for _, val in sorted(wall_psnr_ndt.items(), reverse=True)], width=width, color=colors[0])
psnr_ax.bar(idx, [val for _, val in sorted(copier_psnr_ndt.items(), reverse=True)], width=width, color=colors[1])
psnr_ax.bar(idx + width, [val for _, val in sorted(plant_psnr_ndt.items(), reverse=True)], width=width, color=colors[2])

err_path_wall = os.path.join(results_path, 'stonewall_ndtmap/recon_mean.csv')
err_path_copier = os.path.join(results_path, 'copyroom_ndtmap/recon_mean.csv')
err_path_plant = os.path.join(results_path, 'lounge_ndtmap/recon_mean.csv')

wall_err_ndt = read_from_csv(err_path_wall)
copier_err_ndt = read_from_csv(err_path_copier)
plant_err_ndt = read_from_csv(err_path_plant)

idx = np.asarray([i for i in range(2, 4)])
err_ax.bar(idx - width, [val for _, val in sorted(wall_err_ndt.items(), reverse=True)], width=width, color=colors[0])
err_ax.bar(idx, [val for _, val in sorted(copier_err_ndt.items(), reverse=True)], width=width, color=colors[1])
err_ax.bar(idx + width, [val for _, val in sorted(plant_err_ndt.items(), reverse=True)], width=width, color=colors[2])

mem_path_wall = os.path.join(results_path, 'stonewall_ndtmap/mem.csv')
mem_path_copier = os.path.join(results_path, 'copyroom_ndtmap/mem.csv')
mem_path_plant = os.path.join(results_path, 'lounge_ndtmap/mem.csv')

wall_mem_ndt = read_from_csv(mem_path_wall)
copier_mem_ndt = read_from_csv(mem_path_copier)
plant_mem_ndt = read_from_csv(mem_path_plant)

idx = np.asarray([i for i in range(2, 4)])
mem_ax.bar(idx - width, [val / 1e6 for _, val in sorted(wall_mem_ndt.items(), reverse=True)], width=width, color=colors[0])
mem_ax.bar(idx, [val / 1e6 for _, val in sorted(copier_mem_ndt.items(), reverse=True)], width=width, color=colors[1])
mem_ax.bar(idx + width, [val / 1e6 for _, val in sorted(plant_mem_ndt.items(), reverse=True)], width=width, color=colors[2])

xticklabels += ['NDT ' + str(s) for s, _ in sorted(wall_psnr_ndt.items(), reverse=True)]

# FC
psnr_path_wall = os.path.join(results_path, 'stonewall_fixed_comps/psnr.csv')
psnr_path_copier = os.path.join(results_path, 'copyroom_fixed_comps/psnr.csv')
psnr_path_plant = os.path.join(results_path, 'lounge_fixed_comps/psnr.csv')

wall_psnr_fc = read_from_csv(psnr_path_wall)
copier_psnr_fc = read_from_csv(psnr_path_copier)
plant_psnr_fc = read_from_csv(psnr_path_plant)

idx = np.asarray([i for i in range(4, 7)])
psnr_ax.bar(idx - width, [val for _, val in sorted(wall_psnr_fc.items())], width=width, color=colors[0])
psnr_ax.bar(idx, [val for _, val in sorted(copier_psnr_fc.items())], width=width, color=colors[1])
psnr_ax.bar(idx + width, [val for _, val in sorted(plant_psnr_fc.items())], width=width, color=colors[2])

err_path_wall = os.path.join(results_path, 'stonewall_fixed_comps/recon_mean.csv')
err_path_copier = os.path.join(results_path, 'copyroom_fixed_comps/recon_mean.csv')
err_path_plant = os.path.join(results_path, 'lounge_fixed_comps/recon_mean.csv')

wall_err_fc = read_from_csv(err_path_wall)
copier_err_fc = read_from_csv(err_path_copier)
plant_err_fc = read_from_csv(err_path_plant)

idx = np.asarray([i for i in range(4, 7)])
err_ax.bar(idx - width, [val for _, val in sorted(wall_err_fc.items())], width=width, color=colors[0])
err_ax.bar(idx, [val for _, val in sorted(copier_err_fc.items())], width=width, color=colors[1])
err_ax.bar(idx + width, [val for _, val in sorted(plant_err_fc.items())], width=width, color=colors[2])

mem_path_wall = os.path.join(results_path, 'stonewall_fixed_comps/mem.csv')
mem_path_copier = os.path.join(results_path, 'copyroom_fixed_comps/mem.csv')
mem_path_plant = os.path.join(results_path, 'lounge_fixed_comps/mem.csv')

wall_mem_fc = read_from_csv(mem_path_wall)
copier_mem_fc = read_from_csv(mem_path_copier)
plant_mem_fc = read_from_csv(mem_path_plant)

idx = np.asarray([i for i in range(4, 7)])
mem_ax.bar(idx - width, [val / 1e6 for _, val in sorted(wall_mem_fc.items())], width=width, color=colors[0])
mem_ax.bar(idx, [val / 1e6 for _, val in sorted(copier_mem_fc.items())], width=width, color=colors[1])
mem_ax.bar(idx + width, [val / 1e6 for _, val in sorted(plant_mem_fc.items())], width=width, color=colors[2])

xticklabels += ['FC ' + str(int(s)) for s, _ in sorted(wall_psnr_fc.items())]

# SOGMM
psnr_path_wall = os.path.join(results_path, 'stonewall_sogmm/psnr.csv')
psnr_path_copier = os.path.join(results_path, 'copyroom_sogmm/psnr.csv')
psnr_path_plant = os.path.join(results_path, 'lounge_sogmm/psnr.csv')

wall_psnr_sogmm = read_from_csv(psnr_path_wall)
copier_psnr_sogmm = read_from_csv(psnr_path_copier)
plant_psnr_sogmm = read_from_csv(psnr_path_plant)

idx = np.asarray([i for i in range(7, 10)])
psnr_ax.bar(idx - width, [val for _, val in sorted(wall_psnr_sogmm.items(), reverse=True)], width=width, color=colors[0])
psnr_ax.bar(idx, [val for _, val in sorted(copier_psnr_sogmm.items(), reverse=True)], width=width, color=colors[1])
psnr_ax.bar(idx + width, [val for _, val in sorted(plant_psnr_sogmm.items(), reverse=True)], width=width, color=colors[2])

err_path_wall = os.path.join(results_path, 'stonewall_sogmm/recon_mean.csv')
err_path_copier = os.path.join(results_path, 'copyroom_sogmm/recon_mean.csv')
err_path_plant = os.path.join(results_path, 'lounge_sogmm/recon_mean.csv')

wall_err_sogmm = read_from_csv(err_path_wall)
copier_err_sogmm = read_from_csv(err_path_copier)
plant_err_sogmm = read_from_csv(err_path_plant)

idx = np.asarray([i for i in range(7, 10)])
err_ax.bar(idx - width, [val for _, val in sorted(wall_err_sogmm.items(), reverse=True)], width=width, color=colors[0])
err_ax.bar(idx, [val for _, val in sorted(copier_err_sogmm.items(), reverse=True)], width=width, color=colors[1])
err_ax.bar(idx + width, [val for _, val in sorted(plant_err_sogmm.items(), reverse=True)], width=width, color=colors[2])

mem_path_wall = os.path.join(results_path, 'stonewall_sogmm/mem.csv')
mem_path_copier = os.path.join(results_path, 'copyroom_sogmm/mem.csv')
mem_path_plant = os.path.join(results_path, 'lounge_sogmm/mem.csv')

wall_mem_sogmm = read_from_csv(mem_path_wall)
copier_mem_sogmm = read_from_csv(mem_path_copier)
plant_mem_sogmm = read_from_csv(mem_path_plant)

idx = np.asarray([i for i in range(7, 10)])
mem_ax.bar(idx - width, [val / 1e6 for _, val in sorted(wall_mem_sogmm.items(), reverse=True)], width=width, color=colors[0])
mem_ax.bar(idx, [val / 1e6 for _, val in sorted(copier_mem_sogmm.items(), reverse=True)], width=width, color=colors[1])
mem_ax.bar(idx + width, [val / 1e6 for _, val in sorted(plant_mem_sogmm.items(), reverse=True)], width=width, color=colors[2])

xticklabels += ['SOGMM ' + str(s) for s, _ in sorted(wall_psnr_sogmm.items(), reverse=True)]

# wrap-up this figure
psnr_ax.set_xticks(np.asarray([i for i in range(0, 10)]))
psnr_ax.set_xticklabels(xticklabels, rotation=15)
psnr_ax.legend(['Wall', 'Copier', 'Plant'])
psnr_ax.set_ylabel('PSNR')

if tikz_save:
    tz.save(figure=psnr_fig, filepath=os.path.join(paper_fig_path, 'single-view-psnr.tex'))

err_ax.set_xticks(np.asarray([i for i in range(0, 10)]))
err_ax.set_xticklabels(xticklabels, rotation=15)
err_ax.legend(['Wall', 'Copier', 'Plant'])
err_ax.set_ylabel('Recon. Err. (m)')

if tikz_save:
    tz.save(figure=err_fig, filepath=os.path.join(paper_fig_path, 'single-view-err.tex'))

mem_ax.set_xticks(np.asarray([i for i in range(0, 10)]))
mem_ax.set_xticklabels(xticklabels, rotation=15)
mem_ax.legend(['Wall', 'Copier', 'Plant'])
mem_ax.set_ylabel('Memory (MB)')

if tikz_save:
    tz.save(figure=mem_fig, filepath=os.path.join(paper_fig_path, 'single-view-mem.tex'))

plt.show()
