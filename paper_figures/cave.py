import os
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import *
from mean_shift_py import MeanShift as MSf2

def custom_draw_geometry_with_key_callback(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

plt.style.use('dark_background')

results_path = '/Volumes/GoogleDrive/My Drive/phd/adaptive_perception/results'

color = os.path.join(results_path, 'cave/color.png')
depth = os.path.join(results_path, 'cave/depth.png')
K = np.load(os.path.join(results_path, 'cave/intrinsic.npz'))['arr_0']

iu = ImageUtils(K, depth_min=0.2, depth_max=6.5)

pcld, _ = iu.generate_pcld_cf(color, depth, (1280, 720))

d = np.array([np.linalg.norm(x) for x in pcld[:, 0:3]])[:, np.newaxis]
g = pcld[:, 3][:, np.newaxis]
ms_data = np.concatenate((d, g), axis=1)

ms = MSf2(0.05)
ms.fit(ms_data)
n_components = ms.get_num_modes()
print('num components', n_components)
cluster_centers = ms.get_mode_centers()

nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(cluster_centers)
mm_labels = np.zeros(ms_data.shape[0], dtype=int)
distances, idxs = nbrs.kneighbors(ms_data)
mm_labels = idxs.flatten()

fig, ax = plt.subplots()
ax.set_xlim([0.2, 6.5])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel('Depth [0.2, 6.5]')
ax.set_ylabel('Grayscale [0.0, 1.0]')

for k in range(n_components):
    my_members = mm_labels == k
    ax.scatter(ms_data[my_members, 0], ms_data[my_members, 1], s=1, color=np.random.rand(3,))

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=25, marker='o', edgecolors='black')

plt.show()


# with open(os.path.join(results_path, 'cave/cave.pkl'), 'rb') as f:
#     loaded_model = pickle.load(f)

# pts = loaded_model.sample(3*loaded_model.support_size_, 1.8)
# np.savetxt('cave_pts.txt', pts)

# custom_draw_geometry_with_key_callback(np_to_o3d(pts))