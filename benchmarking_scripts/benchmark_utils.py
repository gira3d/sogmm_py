import os
import numpy as np
import open3d as o3d
import glob
import pickle
import pprint
import csv
from termcolor import cprint
from sklearn.mixture import GaussianMixture

from octomap_py import ColorOcTree
from ndt_map import LazyGrid, NDTMap
from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import *

pp = pprint.PrettyPrinter()


class BenchmarkUtils:
    def __init__(self, d, K=None):
        self.df = d

        if K is None:
            self.K = np.eye(3)
            self.K[0, 0] = 525.0/d
            self.K[1, 1] = 525.0/d
            self.K[0, 2] = 319.5/d
            self.K[1, 2] = 239.5/d
        else:
            self.K = K

        self.iu = ImageUtils(self.K)

    def evaluate_gmm_model(self, model, gt_pose, gt_pcd):
        n_samples = gt_pcd.shape[0]
        _, gt_g = self.iu.pcld_wf_to_imgs(gt_pose, gt_pcd)

        sg = SOGMM()
        sg.model = model

        # evaluate recon. error, precision, recall, f-score
        pr_pcd = sg.joint_dist_sample(n_samples)
        f, p, re, rmean, rstd = calculate_depth_metrics(
            np_to_o3d(gt_pcd), np_to_o3d(pr_pcd))

        # evaluate PSNR and SSIM
        regressed_colors = np.zeros((n_samples, 1))
        regressed_colors = sg.color_conditional(gt_pcd[:, 0:3])

        regressed_pcd = np.zeros(gt_pcd.shape)
        regressed_pcd[:, 0:3] = gt_pcd[:, 0:3]
        regressed_pcd[:, 3] = np.squeeze(regressed_colors)

        _, pr_g = self.iu.pcld_wf_to_imgs(gt_pose, regressed_pcd)

        psnr, ssim = calculate_color_metrics(gt_g, pr_g)

        # memory
        M = model.n_components_
        mem_bytes = 4 * M * (1 + 10 + 4)

        return psnr, ssim, f, p, re, rmean, rstd, mem_bytes

    def evaluate_octomap_model(self, model, gt_pose, gt_pcd):
        _, gt_g = self.iu.pcld_wf_to_imgs(gt_pose, gt_pcd)

        # evaluate recon. error, precision, recall, f-score
        pr_pcd = model.get_color_occ_points()
        f, p, re, rmean, rstd = calculate_depth_metrics(
            np_to_o3d(gt_pcd), np_to_o3d(pr_pcd))

        # evaluate PSNR and SSIM
        regressed_colors = model.get_color_at_points(gt_pcd[:, 0:3])
        regressed_pcd = np.zeros(gt_pcd.shape)
        regressed_pcd[:, 0:3] = gt_pcd[:, 0:3]
        regressed_pcd[:, 3] = np.squeeze(regressed_colors)

        _, pr_g = self.iu.pcld_wf_to_imgs(gt_pose, regressed_pcd)

        psnr, ssim = calculate_color_metrics(gt_g, pr_g)

        # memory
        model.write('temp.ot')
        mem_bytes = os.path.getsize('temp.ot')

        return psnr, ssim, f, p, re, rmean, rstd, mem_bytes

    def evaluate_ndtmap_model(self, model, ndt, gt_pose, gt_pcd):
        n_samples = gt_pcd.shape[0]
        _, gt_g = self.iu.pcld_wf_to_imgs(gt_pose, gt_pcd)

        # evaluate recon. error, precision, recall, f-score
        pr_pcd, _ = model.sample(n_samples)
        f, p, re, rmean, rstd = calculate_depth_metrics(
            np_to_o3d(gt_pcd), np_to_o3d(pr_pcd))

        # evaluate PSNR and SSIM
        regressed_pcd = ndt.get_intensity_at_pcld(gt_pcd[:, 0:3])
        _, pr_g = self.iu.pcld_wf_to_imgs(gt_pose, regressed_pcd)
        psnr, ssim = calculate_color_metrics(gt_g, pr_g)

        # memory
        M = model.n_components
        mem_bytes = 4 * M * (1 + 3 + 6)

        return psnr, ssim, f, p, re, rmean, rstd, mem_bytes


class DatasetProcessor:
    def __init__(self, datasets_root, dataset_name, results_root, decimate_factor,
                 size=(640, 480)):
        self.dr = datasets_root
        self.rr = results_root
        self.dn = dataset_name
        self.size = size

        traj_string = os.path.join(self.dr, self.dn + '-traj.log')
        self.traj = read_log_trajectory(traj_string)

        self.bu = BenchmarkUtils(decimate_factor)

    def get_gt_pose(self, frame):
        return self.traj[frame].pose

    def get_pcd_and_images(self, frame, rgb_ext='.png', depth_ext='.png'):
        rgb_path = os.path.join(self.dr, self.dn + '-color/' + frame + rgb_ext)
        depth_path = os.path.join(
            self.dr, self.dn + '-depth/' + frame + depth_ext)
        w = int(self.size[0] / self.bu.df)
        h = int(self.size[1] / self.bu.df)
        pcd, imgs = self.bu.iu.generate_pcld_cf(
            rgb_path=rgb_path, depth_path=depth_path, size=(w, h))

        return pcd, imgs

    def get_local_gt_pcd(self, frame):
        results_path = os.path.join(self.rr, self.dn + '_fixed_comps')
        pcld_gt_path = os.path.join(
            results_path, 'local_pcd_' + str(frame) + '.pcd')
        return o3d_to_np(o3d.io.read_point_cloud(pcld_gt_path, format='pcd'))

    def write_dict_to_csv(self, dict, csv_path, fn):
        with open(csv_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fn)
            writer.writeheader()
            for key, value in dict.items():
                writer.writerow({fn[0]: key, fn[1]: value})

    def write_metrics(self, results_path, results, fn_prefix):
        psnr_path = os.path.join(results_path, 'psnr.csv')
        self.write_dict_to_csv(results[0], psnr_path, fn=[fn_prefix, 'PSNR'])
        ssim_path = os.path.join(results_path, 'ssim.csv')
        self.write_dict_to_csv(results[1], ssim_path, fn=[fn_prefix, 'SSIM'])
        f_path = os.path.join(results_path, 'fscore.csv')
        self.write_dict_to_csv(results[2], f_path, fn=[fn_prefix, 'F-score'])
        p_path = os.path.join(results_path, 'precision.csv')
        self.write_dict_to_csv(results[3], p_path, fn=[fn_prefix, 'Precision'])
        re_path = os.path.join(results_path, 'recall.csv')
        self.write_dict_to_csv(results[4], re_path, fn=[fn_prefix, 'Recall'])
        rmean_path = os.path.join(results_path, 'recon_mean.csv')
        self.write_dict_to_csv(results[5], rmean_path, fn=[
                               fn_prefix, 'Recon. Mean'])
        rstd_path = os.path.join(results_path, 'recon_std.csv')
        self.write_dict_to_csv(results[6], rstd_path, fn=[
                               fn_prefix, 'Recon. Std. Dev.'])
        mem_path = os.path.join(results_path, 'mem.csv')
        self.write_dict_to_csv(results[7], mem_path, fn=[
                               fn_prefix, 'Memory'])

    def process_fixed_comps_baseline(self, gt_pose, gt_pcd, write_csv=False):
        psnr = {}
        ssim = {}
        f = {}
        p = {}
        re = {}
        rmean = {}
        rstd = {}
        mem = {}

        results_path = os.path.join(self.rr, self.dn + '_fixed_comps')
        model_paths = glob.glob(os.path.join(results_path, 'full_model*.pkl'))
        for _, m in enumerate(model_paths):
            with open(m, 'rb') as mf:
                model = pickle.load(mf)

            assert (model.support_size_ == gt_pcd.shape[0])
            comps = int(os.path.splitext(m)[0].split('_')[-1])

            metrics = self.bu.evaluate_gmm_model(model, gt_pose, gt_pcd)

            psnr[comps] = metrics[0]
            ssim[comps] = metrics[1]
            f[comps] = metrics[2]
            p[comps] = metrics[3]
            re[comps] = metrics[4]
            rmean[comps] = metrics[5]
            rstd[comps] = metrics[6]
            mem[comps] = metrics[7]

        cprint('psnr ' + pp.pformat(psnr), 'red')
        cprint('ssim ' + pp.pformat(ssim), 'red')
        cprint('f-score ' + pp.pformat(f), 'red')
        cprint('precision ' + pp.pformat(p), 'red')
        cprint('recall ' + pp.pformat(re), 'red')
        cprint('recon. mean ' + pp.pformat(rmean), 'red')
        cprint('recon. std ' + pp.pformat(rstd), 'red')
        cprint('memory ' + pp.pformat(mem), 'red')

        if write_csv:
            self.write_metrics(
                results_path, (psnr, ssim, f, p, re, rmean, rstd, mem), fn_prefix='M')

        return psnr, ssim, f, p, re, rmean, rstd, mem

    def process_octomap_baseline(self, gt_pose, gt_pcd, resolutions, write_csv=False):
        psnr = {}
        ssim = {}
        f = {}
        p = {}
        re = {}
        rmean = {}
        rstd = {}
        mem = {}

        results_path = os.path.join(self.rr, self.dn + '_octomap')
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        for r in resolutions:
            model = ColorOcTree(r)
            model.insert_color_occ_points(gt_pcd)
            model.update_inner_occupancy()

            metrics = self.bu.evaluate_octomap_model(model, gt_pose, gt_pcd)
            psnr[r] = metrics[0]
            ssim[r] = metrics[1]
            f[r] = metrics[2]
            p[r] = metrics[3]
            re[r] = metrics[4]
            rmean[r] = metrics[5]
            rstd[r] = metrics[6]
            mem[r] = metrics[7]

        cprint('psnr ' + pp.pformat(psnr), 'red')
        cprint('ssim ' + pp.pformat(ssim), 'red')
        cprint('f-score ' + pp.pformat(f), 'red')
        cprint('precision ' + pp.pformat(p), 'red')
        cprint('recall ' + pp.pformat(re), 'red')
        cprint('recon. mean ' + pp.pformat(rmean), 'red')
        cprint('recon. std ' + pp.pformat(rstd), 'red')
        cprint('memory ' + pp.pformat(mem), 'red')

        if write_csv:
            self.write_metrics(
                results_path, (psnr, ssim, f, p, re, rmean, rstd, mem), fn_prefix='Resolution')

        return psnr, ssim, f, p, re, rmean, rstd, mem

    def process_ndtmap_baseline(self, gt_pose, gt_pcd, resolutions, write_csv=False):
        psnr = {}
        ssim = {}
        f = {}
        p = {}
        re = {}
        rmean = {}
        rstd = {}
        mem = {}

        results_path = os.path.join(self.rr, self.dn + '_ndtmap')
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        for r in resolutions:
            l = LazyGrid(r)
            n = NDTMap(l)
            n.load_pointcloud(gt_pcd)
            n.compute_ndt_cells_simple()

            weights, means, covs = n.get_gaussians()
            weights /= np.sum(weights)
            n_components = np.shape(weights)[0]

            model = GaussianMixture(n_components=n_components, covariance_type='full')
            model.weights_ = weights
            model.means_ = means
            model.covariances_ = matrix_to_tensor(covs, 3)

            metrics = self.bu.evaluate_ndtmap_model(model, n, gt_pose, gt_pcd)
            psnr[r] = metrics[0]
            ssim[r] = metrics[1]
            f[r] = metrics[2]
            p[r] = metrics[3]
            re[r] = metrics[4]
            rmean[r] = metrics[5]
            rstd[r] = metrics[6]
            mem[r] = metrics[7]

        cprint('psnr ' + pp.pformat(psnr), 'red')
        cprint('ssim ' + pp.pformat(ssim), 'red')
        cprint('f-score ' + pp.pformat(f), 'red')
        cprint('precision ' + pp.pformat(p), 'red')
        cprint('recall ' + pp.pformat(re), 'red')
        cprint('recon. mean ' + pp.pformat(rmean), 'red')
        cprint('recon. std ' + pp.pformat(rstd), 'red')
        cprint('memory ' + pp.pformat(mem), 'red')

        if write_csv:
            self.write_metrics(
                results_path, (psnr, ssim, f, p, re, rmean, rstd, mem), fn_prefix='Resolution')

        return psnr, ssim, f, p, re, rmean, rstd, mem

    def process_sogmm_proposed(self, gt_pose, gt_pcd, write_csv=False):
        est_comps = {}
        psnr = {}
        ssim = {}
        f = {}
        p = {}
        re = {}
        rmean = {}
        rstd = {}
        mem = {}

        results_path = os.path.join(self.rr, self.dn + '_sogmm')
        model_paths = glob.glob(os.path.join(results_path, '*/full_model.pkl'))
        for _, m in enumerate(model_paths):
            with open(m, 'rb') as mf:
                model = pickle.load(mf)

            assert (model.support_size_ == gt_pcd.shape[0])

            b = float(os.path.splitext(m)[0].split('/')[-2].split('_')[-1])

            est_comps[b] = model.n_components_

            metrics = self.bu.evaluate_gmm_model(model, gt_pose, gt_pcd)
            psnr[b] = metrics[0]
            ssim[b] = metrics[1]
            f[b] = metrics[2]
            p[b] = metrics[3]
            re[b] = metrics[4]
            rmean[b] = metrics[5]
            rstd[b] = metrics[6]
            mem[b] = metrics[7]

        cprint('psnr ' + pp.pformat(psnr), 'green')
        cprint('ssim ' + pp.pformat(ssim), 'green')
        cprint('f-score ' + pp.pformat(f), 'green')
        cprint('precision ' + pp.pformat(p), 'green')
        cprint('recall ' + pp.pformat(re), 'green')
        cprint('recon. mean ' + pp.pformat(rmean), 'green')
        cprint('recon. std ' + pp.pformat(rstd), 'green')
        cprint('memory ' + pp.pformat(mem), 'red')

        if write_csv:
            self.write_metrics(
                results_path, (psnr, ssim, f, p, re, rmean, rstd, mem), fn_prefix='Bandwidth')
            est_comps_path = os.path.join(results_path, 'est_comps.csv')
            self.write_dict_to_csv(
                est_comps, est_comps_path, fn=['Bandwidth', 'M'])

        return psnr, ssim, f, p, re, rmean, rstd, mem
