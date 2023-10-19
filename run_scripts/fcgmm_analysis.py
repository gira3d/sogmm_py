import os
import pickle
import pandas as pd
import open3d as o3d
from cprint import cprint
from config_parser import ConfigParser

from sogmm_cpu import SOGMMInference as CPUInference

from sogmm_py.utils import *

def comp_validity_check(cov):
    # if two smallest semi-axis lengths are smaller than a threshold, ignore the component
    eigvals, _ = np.linalg.eigh(cov)
    sorted_eigvals = np.sort(eigvals)

    if np.sqrt(sorted_eigvals[0]) < 1e-4 and np.sqrt(sorted_eigvals[1]) < 1e-4:
        return False
    else:
        return True

parser = ConfigParser()
parser.add_argument('--config', is_config_file=True)
parser.add_argument('--voxel_size', type=float, default=0.01)
parser.add_argument('--sigma', type=float, default=1.8)

args = parser.get_config()

cases = [100, 200, 400, 800]
pkl_files = [os.path.join(args.path_results, f"fcgmm_results/{args.dataset_name}_model_{str(a)}_cpu.pkl") for a in cases]
gt_file = os.path.join(args.path_results, f"ground_truth/{args.dataset_name}_gt_voxelized.pcd")
gt_pcld_downsampled = o3d.io.read_point_cloud(gt_file)
infer = CPUInference()

metrics = np.zeros((4, 6))
for i, p in enumerate(pkl_files):
    if os.path.isfile(p):
        sogmm = None
        with open(p, 'rb') as f:
            sogmm = pickle.load(f)

        batch_size = 100
        comp_list = list(np.arange(sogmm.n_components_))

        covariances = matrix_to_tensor(sogmm.covariances_, 4)
        pr_pcds = []
        pr_Ns = []
        for k in range(0, sogmm.n_components_, batch_size):
            sub_comps = comp_list[k:k+batch_size]
            valid = [comp_validity_check(covariances[a])
                        for a in sub_comps]
            submap = sogmm.submap_from_indices(
                [sub_comps[a] for a in range(len(sub_comps)) if valid[a] is True])

            pcld = infer.reconstruct_fast(submap, submap.support_size_, args.sigma)
            pr_pcds.append(pcld)
            pr_Ns.append(pcld.shape[0])

        full_pr_pcld = np.zeros((sum(pr_Ns), 4))
        prev_i = 0
        for j in range(0, len(pr_pcds)):
            n = pr_Ns[j]
            assert (n == pr_pcds[j].shape[0])
            full_pr_pcld[prev_i:prev_i+n, :] = pr_pcds[j]
            prev_i += n

        pr_pcld_downsampled = np_to_o3d(full_pr_pcld).voxel_down_sample(voxel_size=args.voxel_size)

        pr_pcld_file = os.path.join(args.path_results, f"fcgmm_results/{args.dataset_name}_{str(cases[i])}_pr_voxelized.pcd")
        o3d.io.write_point_cloud(pr_pcld_file, pr_pcld_downsampled)

        result = calculate_all_metrics(gt_pcld_downsampled, pr_pcld_downsampled, th=args.voxel_size)

        metrics[i, 0] = cases[i]
        metrics[i, 1] = result[0]
        metrics[i, 2] = result[1]
        metrics[i, 3] = result[2]
        metrics[i, 4] = result[3]
        metrics[i, 5] = result[4]
        cprint.ok("Processed %s" % (p))

for l in range(len(cases)):
    cprint.info(f"comps {metrics[l, 0]}\n"
                "--------------------------\n"
                f"f-score {(metrics[l, 1]):.2f}\n"
                f"precision {(metrics[l, 2]):.2f}\n"
                f"recall {(metrics[l, 3]):.2f}\n"
                f"mre {(metrics[l, 4]):.4f}\n"
                f"psnr {(metrics[l, 5]):.2f}\n")

df = pd.DataFrame(metrics, columns=['comps',
                                    'f-score',
                                    'precision',
                                    'recall',
                                    'mre',
                                    'psnr'])
df.set_index('comps')
metrics_file = os.path.join(args.path_results, f"fcgmm_results/{args.dataset_name}_fcgmm_recon_metrics.csv")
df.to_csv(metrics_file)