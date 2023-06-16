import argparse
from termcolor import cprint

from benchmark_utils import DatasetProcessor

parser = argparse.ArgumentParser(description="SOGMM system")
parser.add_argument('--datasetname', type=str)
parser.add_argument('--datasetroot', type=str)
parser.add_argument('--decimate', type=int)
parser.add_argument('--resultsroot', type=str)
parser.add_argument('--frame', type=int)

args = parser.parse_args()

results_root = args.resultsroot
dataset_name = args.datasetname
datasets_root = args.datasetroot
decimate_factor = args.decimate
frame = args.frame

data_process = DatasetProcessor(datasets_root, dataset_name,
                                results_root, decimate_factor)

gt_pose = data_process.get_gt_pose(frame)
gt_pcd = data_process.get_local_gt_pcd(frame)

# baseline 1: fixed number of components
cprint('processing baseline 1: fixed number of components', 'grey')
data_process.process_fixed_comps_baseline(gt_pose, gt_pcd, write_csv=True)

# baseline 2: octomap
cprint('processing baseline 2: octomap', 'grey')
data_process.process_octomap_baseline(gt_pose, gt_pcd, resolutions=[0.05, 0.02], write_csv=True)

# baseline 3: ndtmap
cprint('processing baseline 3: ndtmap', 'grey')
data_process.process_ndtmap_baseline(gt_pose, gt_pcd, resolutions=[0.05, 0.02], write_csv=True)

# ours: SOGMM
cprint('processing ours: SOGMM', 'grey')
data_process.process_sogmm_proposed(gt_pose, gt_pcd, write_csv=True)