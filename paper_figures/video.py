import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import *

def custom_draw_geometry_with_key_callback(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0.75, 0.0)
        return False

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.point_size = 2
        opt.background_color = np.asarray([0, 0, 0])
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = rotate_view
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

pcld = o3d.io.read_point_cloud('/media/fractal/T7/rss2023-resub/results/isogmm_results/livingroom1_bw_0_02_pr.pcd')
pcld_np = o3d_to_np(pcld)
pcld_np_cropped = pcld_np[pcld_np[:, 1] < 2.6, :]
pcld_np_cropped_2 = pcld_np_cropped[pcld_np_cropped[:, 2] < 3.1, :]
pcld_cropped_2 = np_to_o3d(pcld_np_cropped_2)

custom_draw_geometry_with_key_callback(pcld_cropped_2)