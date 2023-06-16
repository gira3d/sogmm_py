"""Utilities for Self-Organizing Gaussian Mixture Models."""

import os
import cv2
from cv2 import INTER_NEAREST
import numpy as np
import open3d as o3d
import numbers
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .liegroups.numpy.se3 import SE3Matrix

from camera_model_py import CameraModel


class RedwoodCameraPose(object):
    """Container for camera poses as specified by Open3D/ICL-NUIM dataset.

    Attributes
    ----------
    metadata : map(int, list(str))
        Metadata for the image indices etc. corresponding to this pose
    pose : array, shape (4, 4)
        SE(3) pose
    """

    def __init__(self, meta, mat):
        """
        Parameters
        ----------
        meta : map(int, list(str))
            Metadata for the image indices etc. corresponding to this pose
        mat : array, shape (4, 4)
            SE(3) pose
        """

        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)


class ImageUtils(object):
    """Utilities for RGB-D images and point cloud conversions.

    Utilizes Open3D to create and convert images to and from multi- and
    single-modal point clouds.

    Attributes
    ----------
    camera_model : camera_model_py.CameraModel
        Pinhole camera model class defined in C++ (accessible through pybind11).
    depth_scale : float
        Scaling parameter for the depth data.
    depth_min : float
        Minimum relevant depth.
    depth_max : float
        Maximum relevant depth.
    """

    def __init__(self, intrisics_matrix,
                 depth_scale=1000.0,
                 depth_min=0.1,
                 depth_max=3.0,
                 im_w=640,
                 im_h=480):
        """
        Parameters
        ----------
        intrisics_matrix : array-like, shape (3, 3)
            Intrinsics matrix for the pinhole camera model.
        depth_scale : float, optional (default: 1000.0)
            Scaling for the depth image.
        depth_min : float, optional (default: 0.1)
            Minimum relevant depth.
        depth_max : float, optional (default: 3.0)
            Maximum relevant depth.
        im_w : int, optional (default: 640)
            Camera image width.
        im_h : int, optional (default: 480)
            Camera image height.
        """

        self.camera_model = CameraModel(intrisics_matrix, im_w, im_h)
        self.depth_scale = depth_scale
        self.depth_min = depth_min
        self.depth_max = depth_max

    def read_dg_img(self, rgb_path, depth_path, size):
        """Read depth-grayscale image via Open3D I/O.

        Parameters
        ----------
        rgb_path : str
            Path to the rgb image file.
        depth_path : str
            Path to the depth image file.
        size : (width, height)
            Image width and height if resizing is desired.

        Returns
        -------
        resized_grayscale_arr : array, shape (height, width)
            Numpy array for the resized grayscale image.
        resized_depth_arr : array, shape (height, width)
            Numpy array for the resized depth image.
        rgbd_im : open3d.geometry.RGBDImage
            Open3D geometry that can be used for visualization, etc.
        """

        color_raw = o3d.io.read_image(rgb_path)
        depth_raw = o3d.io.read_image(depth_path)
        rgbd_im = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, depth_scale=self.depth_scale, depth_trunc=self.depth_max)
        grayscale_arr = np.asarray(rgbd_im.color)
        depth_arr = np.asarray(rgbd_im.depth)
        resized_grayscale_arr = cv2.resize(grayscale_arr,
                                           size,
                                           interpolation=INTER_NEAREST)
        resized_depth_arr = cv2.resize(depth_arr,
                                       size,
                                       interpolation=INTER_NEAREST)
        return resized_grayscale_arr, resized_depth_arr, rgbd_im

    def generate_pcld_cf(self, rgb_path, depth_path, size):
        """Generate depth-grayscale point cloud in the camera frame.

        Parameters
        ----------
        rgb_path : str
            Path to the rgb image file.
        depth_path : str
            Path to the depth image file.
        size : (width, height)
            Image width and height if resizing is desired.

        Returns
        -------
        pcld : array-like, shape (width * height, 4)
            Point cloud with \\( x, y, z, g \\) data in the camera frame.
        rgbd_im : open3d.geometry.RGBDImage
            Open3D geometry that can be used for visualization, etc.
        """

        g_arr, d_arr, rgbd_im = self.read_dg_img(rgb_path, depth_path, size)

        # project to 3D and get the valid indices within min. and max. depth
        d_pcld, valid_idxs = self.camera_model.to_3d(d_arr,
                                                     self.depth_min,
                                                     self.depth_max)
        # allow all grayscale values by using (-1.0, 100.0) as the range
        g_pcld, _ = self.camera_model.to_3d(g_arr, -1.0, 100.0)

        # filter the point clouds with the valid indices
        d_pcld = d_pcld[valid_idxs, :]
        g_pcld = g_pcld[valid_idxs, :]

        # concatenate and return
        pcld = np.zeros((d_pcld.shape[0], 4))
        pcld[:, 0:3] = d_pcld[:, 0:3]
        pcld[:, 3] = g_pcld[:, 2]

        return pcld, rgbd_im

    def generate_pcld_wf(self, Tw, rgb_path, depth_path, size):
        """Generate depth-grayscale point cloud in the world frame.

        Parameters
        ----------
        Tw : array-like, shape (4, 4)
            Transformation from camera frame to the world frame.
        rgb_path : str
            Path to the rgb image file.
        depth_path : str
            Path to the depth image file.
        size : (width, height)
            Image width and height if resizing is desired.

        Returns
        -------
        pcld : array-like, shape (width * height, 4)
            Point cloud with \\( x, y, z, g \\) data in the world frame.
        rgbd_im : open3d.geometry.RGBDImage
            Open3D geometry that can be used for visualization, etc.
        """

        # get camera frame point cloud
        pcld_cf, rgbd_im = self.generate_pcld_cf(rgb_path, depth_path, size)
        n_points = pcld_cf.shape[0]

        # convert the depth part into homogeneous coordinates
        d_pcld_cf_h = np.concatenate(
            (pcld_cf[:, 0:3], np.ones((n_points, 1))), axis=1)

        # transform the depth part to world frame
        Twse3 = SE3Matrix.from_matrix(Tw, normalize=True)
        d_pcld_wf_h = Twse3.dot(d_pcld_cf_h)
        d_pcld_wf = np.delete(d_pcld_wf_h, 3, axis=1)

        # concatenate and return
        pcld = np.zeros((n_points, 4))
        pcld[:, 0:3] = d_pcld_wf
        pcld[:, 3] = pcld_cf[:, 3]

        return pcld, rgbd_im

    def pcld_wf_to_imgs(self, Tw, pcld, gt=True):
        """Convert world frame point cloud to image using this camera model.
        """
        n_points = pcld.shape[0]

        Twse3 = SE3Matrix.from_matrix(Tw, normalize=True)
        d_pcld_wf_h = np.concatenate(
            (pcld[:, 0:3], np.ones((n_points, 1))), axis=1)
        d_pcld_cf_h = Twse3.inv().dot(d_pcld_wf_h)
        d_pcld_cf = np.delete(d_pcld_cf_h, 3, axis=1)
        pcld_cf = np.zeros(pcld.shape)
        pcld_cf[:, 0:3] = d_pcld_cf
        pcld_cf[:, 3] = pcld[:, 3]

        depth_im, _ = self.camera_model.to_2d(d_pcld_cf, gt)
        grayscale_im, _ = self.camera_model.to_2d_dim(pcld_cf, 3, gt)

        return depth_im, grayscale_im

    def generate_inc_pcld_wf(self, Tw1, Tw2, rgb_path, depth_path, size):
        Tw1se3 = SE3Matrix.from_matrix(Tw1)
        T1wse3 = Tw1se3.inv()
        Tw2se3 = SE3Matrix.from_matrix(Tw2)
        T12se3 = (T1wse3.dot(Tw2se3)).as_matrix()

        # get world frame pcld for frame 2
        pcld_cf, _ = self.generate_pcld_cf(rgb_path, depth_path, size)
        n_points = pcld_cf.shape[0]
        d_pcld_cf_h = np.concatenate(
            (pcld_cf[:, 0:3], np.ones((n_points, 1))), axis=1)

        # project to frame 1
        d_pcld1_h = (np.dot(T12se3, d_pcld_cf_h.transpose())).transpose()
        d_pcld1 = np.delete(d_pcld1_h, 3, axis=1)
        _, inv_idxs = self.camera_model.to_2d(d_pcld1, False)

        d_pcld2_hw = Tw2se3.dot(d_pcld_cf_h[inv_idxs, :])
        if len(inv_idxs) == 1:
            d_pcld2_hw = np.expand_dims(d_pcld2_hw, axis=0)
        d_pcld2_w = np.delete(d_pcld2_hw, 3, axis=1)

        novel_pcld = np.zeros((len(inv_idxs), 4))
        novel_pcld[:, 0:3] = d_pcld2_w[:, 0:3]
        novel_pcld[:, 3] = pcld_cf[inv_idxs, 3]
        
        return novel_pcld


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def read_log_trajectory(filename):
    """Read trajectory from a .log file.

    Only works for ICL-NUIM dataset.

    Parameters
    ----------
    filename : str
        Path to the trajectory file.

    Returns
    -------
    traj : list(RedwoodCameraPose)
        List of SE(3) poses with metadata
    """

    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline()
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
            traj.append(RedwoodCameraPose(metadata, mat))
            metastr = f.readline()
    return traj


def np_to_o3d(np_pcld, color=None):
    """Numpy array to Open3D point cloud conversion. 

    Parameters
    ----------
    np_pcld : array, shape (n_samples, n_features)
        Input numpy point cloud. Note that `n_features` can be either 3, 4, or 6.

    Returns
    -------
    o3d_pcld : open3d.geometry.PointCloud
        Converted open3d point cloud
    """

    n_features = np_pcld.shape[1]

    o3d_pcld = o3d.geometry.PointCloud()
    o3d_pcld.points = o3d.utility.Vector3dVector(np_pcld[:, 0:3])

    if n_features == 3:
        # no color data; paint the pcld with some constant color
        if color is None:
            o3d_pcld.paint_uniform_color([1, 0.706, 0])
        else:
            o3d_pcld.paint_uniform_color(color)
        return o3d_pcld

    if n_features == 4:
        # single modal color data; usually grayscale or thermal
        o3d_pcld.colors = o3d.utility.Vector3dVector(np.concatenate((np_pcld[:, 3][:, np.newaxis],
                                                                     np_pcld[:, 3][:,
                                                                                   np.newaxis],
                                                                     np_pcld[:, 3][:, np.newaxis]), axis=1))
        return o3d_pcld

    if n_features == 6:
        # full color information has been provided
        o3d_pcld.colors = o3d.utility.Vector3dVector(np_pcld[:, 3:6])
        return o3d_pcld


def o3d_to_np(o3d_pcld):
    """Open3D point cloud to 4D numpy array conversion.

    Parameters
    ----------
    o3d_pcld : open3d.geometry.PointCloud
        Converted open3d point cloud

    Returns
    -------
    np_pcld : array, shape (n_samples, n_features)
        Output numpy point cloud. Note that `n_features` here are fixed to 4.
    """

    xyz = np.asarray(o3d_pcld.points)
    colors = np.asarray(o3d_pcld.colors)

    ret = np.zeros((xyz.shape[0], 4))
    ret[:, 0:3] = xyz
    ret[:, 3] = colors[:, 0]

    return ret


def calculate_depth_metrics(gt, pr, th=0.01):
    """Calculate depth reconstruction metrics.

    Parameters
    ----------
    gt : open3d.geometry.PointCloud
        Ground truth Open3D point cloud
    pr : open3d.geometry.PointCloud
        Reconstructed/Predicted Open3d point cloud
    th : float
        Minimum relevant point cloud to mesh distance

    Returns
    -------
    fscore : float
        F-score of reconstruction.
    precision : float
        Precision of reconstruction.
    recall : float
        Recall of reconstruction.
    recon_err_mean : float
        Mean reconstruction error.
    recon_err_std : float
        Std. Dev. reconstruction error.
    """
    # closest dist for each gt point
    d1 = gt.compute_point_cloud_distance(pr)
    # reconstruction error is the mean distance of each 3D point
    recon_err_mean = np.mean(np.asarray(d1))
    recon_err_std = np.std(np.asarray(d1))
    # closest dist for each pred point
    d2 = pr.compute_point_cloud_distance(gt)
    if len(d1) and len(d2):
        # how many of our sampled points lie close to a gt point?
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        # how many of ground truth points are matched?
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall, recon_err_mean, recon_err_std


def calculate_color_metrics(gt, pr):
    """Calculate color reconstruction metrics.

    In this case, gt and pr need to be images of same shape

    Parameters
    ----------
    gt : array, shape (width, height)
        Ground truth image
    pr : array, shape (width, height)
        Reconstructed/Predicted image

    Returns
    -------
    psnr : float
        Peak Signal-to-Noise Ratio.
    ssim : float
        Structural similarity.
    """

    psnr = peak_signal_noise_ratio(gt, pr)
    ssim = structural_similarity(gt, pr, data_range=1.0)
    return psnr, ssim


def tensor_to_matrix(tensor, dim):
    """Flatten a tensor to matrix (used for covariance handling)."""
    n_components, _, _ = tensor.shape
    matrix = np.zeros((n_components, dim * dim))
    for i in range(n_components):
        matrix[i, :] = tensor[i, :, :].flatten(order='C')

    return matrix


def matrix_to_tensor(matrix, dim):
    """Inflate a matrix to tensor (used for covariance handling)."""
    n_components, _ = matrix.shape
    tensor = np.zeros((n_components, dim, dim))
    for i in range(n_components):
        tensor[i, :, :] = matrix[i, :].reshape((dim, dim))

    return tensor

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)