import os
import open3d as o3d
import glob
import numpy as np
import matplotlib.pyplot as plt

from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import ImageUtils, read_log_trajectory, np_to_o3d, calculate_color_metrics

class ICLNUIMCamera:
    def __init__(self, df=1.0):
        # intrinsic matrix without decimation
        K = np.eye(3)
        K[0, 0] = 525.0/df
        K[1, 1] = 525.0/df
        K[0, 2] = 319.5/df
        K[1, 2] = 239.5/df

        # maximum resolution of the images in the dataset
        self.W = (int) (640 / df)
        self.H = (int) (480 / df)

        # operator for full image
        self.iu = ImageUtils(self.K)


class ICLNUIMDataFrame:
    def __init__(self, f, paths, pose, rgbd):
        self.frame = f
        self.paths = paths
        self.pose = pose

        self.rgbd = rgbd


class ICLNUIMDataSet:
    def __init__(self, path, name, rgb_ext, dpth_ext='.png'):
        # meta data
        self.path = path
        self.rgb_ext = rgb_ext
        self.dpth_ext = dpth_ext
        self.name = name

        # paths to depth and rgb files
        self.depth_paths = sorted(glob.glob(os.path.join(self.path,
                                                         self.name + '-depth/*'
                                                         + dpth_ext)))
        self.rgb_paths = sorted(glob.glob(os.path.join(self.path,
                                                       self.name + '-color/*'
                                                       + rgb_ext)))

        # all the ground truth poses for the above depth and rgb images
        self.traj = read_log_trajectory(
            os.path.join(self.path, self.name + '-traj.log'))


    def get_frame(self):
        pass