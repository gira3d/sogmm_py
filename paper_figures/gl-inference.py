import os
import json
import argparse
import numpy as np
import open3d as o3d

from sogmm_py.vis_open3d import VisOpen3D
from sogmm_py.utils import np_to_o3d, o3d_to_np

def vis_nvblox_mesh(path, view_string, target_file):
    vis = VisOpen3D(visible=False)
    render_ops = vis.get_render_option()
    render_ops.point_size = 1
    pcld = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    vis.add_geometries([pcld])
    vis.set_view_status(view_string)
    vis.capture_screen_image(target_file)

def vis_pcd(path, view_string, target_file):
    vis = VisOpen3D(visible=True)
    render_ops = vis.get_render_option()
    render_ops.point_size = 1

    pcld = o3d.io.read_point_cloud(path)
    pcld_np = o3d_to_np(pcld)
    pcld_np_cropped = pcld_np[pcld_np[:, 1] < 2.6, :]
    pcld_cropped = np_to_o3d(pcld_np_cropped)

    vis.add_geometries([pcld_cropped])
    vis.set_view_status(view_string)
    vis.run()

def vis_gmm_pcd(path, view_string, target_file):
    vis = VisOpen3D(visible=False)
    render_ops = vis.get_render_option()
    render_ops.point_size = 1

    pcld = o3d.io.read_point_cloud(path)
    pcld_np = o3d_to_np(pcld)
    pcld_np_cropped = pcld_np[pcld_np[:, 1] < 2.6, :]
    pcld_cropped = np_to_o3d(pcld_np_cropped)

    vis.add_geometries([pcld_cropped])
    vis.set_view_status(view_string)
    vis.capture_screen_image(target_file)

def vis_octomap_pcd(path, view_string, target_file, v):
    vis = VisOpen3D(visible=False)
    render_ops = vis.get_render_option()
    render_ops.point_size = 1

    pcld = o3d.io.read_point_cloud(path)
    pcld_np = o3d_to_np(pcld)
    pcld_np_cropped = pcld_np[pcld_np[:, 1] < 2.6, :]
    pcld_cropped = np_to_o3d(pcld_np_cropped)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcld_cropped, voxel_size=v)

    vis.add_geometries([voxel_grid])
    vis.set_view_status(view_string)
    vis.capture_screen_image(target_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="isogmm metrics")
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--dataset_name', type=str, default="livingroom1")

    args = parser.parse_args()

    livingroom_view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 2.583730936050415, 2.5999975204467773, 4.2289800643920898 ],
			"boundingbox_min" : [ -2.6225473880767822, 0.045030891895294189, -1.9846775531768799 ],
			"field_of_view" : 60.0,
			"front" : [ 0.42010601103967038, 0.8521088750580561, 0.3121240210807727 ],
			"lookat" : [ -0.019408226013183594, 1.3225142061710358, 1.122151255607605 ],
			"up" : [ -0.68270185199437294, 0.52336334468660561, -0.50991076741122343 ],
			"zoom" : 2.0
		}
	],
	"version_major" : 1,
	"version_minor" : 0
    }

    lounge_view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 2.583730936050415, 2.5999975204467773, 4.2289800643920898 ],
			"boundingbox_min" : [ -2.6225473880767822, 0.045030891895294189, -1.9846775531768799 ],
			"field_of_view" : 60.0,
			"front" : [ -0.085948154700654367, -0.92242139092228226, -0.37649925932537326 ],
			"lookat" : [ 2.0916572950719075, 1.2422129553584436, 1.1062828807923488 ],
			"up" : [ -0.015645493914887437, -0.37660141707073896, 0.92624326781924338 ],
			"zoom" : 0.51799999999999868
		}
	],
	"version_major" : 1,
	"version_minor" : 0
    }

    if args.dataset_name == "livingroom1":
        view_string = json.dumps(livingroom_view)
    if args.dataset_name == "lounge":
        view_string = json.dumps(lounge_view)

    vis_gmm_pcd(os.path.join(args.results_path, f'ground_truth/{args.dataset_name}_gt_voxelized.pcd'), view_string,
                os.path.join(args.results_path, f'ground_truth/{args.dataset_name}_gt_voxelized.png'))

    for s in ["0_02", "0_03", "0_04", "0_05"]:
        vis_gmm_pcd(os.path.join(args.results_path, f'isogmm_results/{args.dataset_name}_bw_{s}_pr.pcd'), view_string,
                    os.path.join(args.results_path, f'isogmm_results/{args.dataset_name}_bw_{s}_pr.png'))

    for m in ["800", "400", "200", "100"]:
        vis_gmm_pcd(os.path.join(args.results_path, f'fcgmm_results/{args.dataset_name}_{m}_pr_voxelized.pcd'), view_string,
                    os.path.join(args.results_path, f'fcgmm_results/{args.dataset_name}_{m}_pr_voxelized.png'))

    for r in ["0_02", "0_04", "0_06", "0_08"]:
        vis_octomap_pcd(os.path.join(args.results_path, f'octomap_results/{args.dataset_name}_{r}_octomap.pcd'), view_string,
                        os.path.join(args.results_path, f'octomap_results/{args.dataset_name}_{r}_octomap.png'), float(r.replace('_', '.')))

    
    for r in ["0.02", "0.04", "0.06", "0.08"]:
        vis_nvblox_mesh(os.path.join(args.results_path, f'nvblox_results_2/{args.dataset_name}_{r}_nvblox.ply'), view_string,
                        os.path.join(args.results_path, f'nvblox_results_2/{args.dataset_name}_{r}_nvblox.png'))
