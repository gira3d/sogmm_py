import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import *

def custom_draw_geometry_with_key_callback(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

results_path = '/Volumes/GoogleDrive/My Drive/phd/adaptive_perception/results'

with open(os.path.join(results_path, 'cave/cave.pkl'), 'rb') as f:
    loaded_model = pickle.load(f)

pts = loaded_model.sample(3*loaded_model.support_size_, 1.8)
np.savetxt('cave_pts.txt', pts)

pcd = np_to_o3d(pts)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
# colors = plt.get_cmap("inferno")(pts[:, 3])
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

custom_draw_geometry_with_key_callback(pcd)