from scipy.io import loadmat
import matplotlib.pyplot as plt
import pickle
from ndt_map import LazyGrid, NDTMap
from sklearn.mixture import GaussianMixture

from sogmm_py.sogmm import SOGMM
from mean_shift_py import MeanShift as MSf2
from sogmm_py.utils import *
from benchmark_utils import BenchmarkUtils

xyz_path = '/Users/kshitijgoel/Documents/main/code.nosync/thermogauss/xyz_bremen.mat'
thermal_path = '/Users/kshitijgoel/Documents/main/code.nosync/thermogauss/thermal_bremen.mat'

xyz = loadmat(xyz_path)['all'] / 100
xyz[:, 2] += np.abs(np.min(xyz[:, 2]))
thermal = loadmat(thermal_path)['thermals']

thermal_max = np.max(thermal)
thermal_min = np.min(thermal)
print('thermal max %f min %f' % (thermal_max, thermal_min))

norm_thermal = (thermal - thermal_min) / (thermal_max - thermal_min)
assert (np.max(norm_thermal) == 1)
assert (np.min(norm_thermal) == 0)

pcld = np.concatenate((xyz, norm_thermal), axis=1)
assert (pcld.shape == (xyz.shape[0], 4))

K = np.eye(3)
K[0, 0] = 525.0
K[1, 1] = 525.0
K[0, 2] = 319.5
K[1, 2] = 239.5
iu = ImageUtils(K)
iu.camera_model.w = 640
iu.camera_model.h = 480

Twb = np.eye(4)
Twb[:3, :3] = np.array([[-0.006202392727880,  -0.029074948926852,  0.999557991148763],
                        [-0.999780767796089,  0.020170897790247,  -0.005617047925494],
                        [-0.019998666693333,  -0.999373694984632,  -0.029193682591489]])
Twb[0, 3] = -50.0
Twb[1, 3] = -90.0
Twb[2, 3] = 15.0

n_points = pcld.shape[0]

Twse3 = SE3Matrix.from_matrix(Twb, normalize=True)
d_pcld_wf_h = np.concatenate(
    (pcld[:, 0:3], np.ones((n_points, 1))), axis=1)
d_pcld_cf_h = Twse3.inv().dot(d_pcld_wf_h)
d_pcld_cf = np.delete(d_pcld_cf_h, 3, axis=1)
pcld_cf = np.zeros(pcld.shape)
pcld_cf[:, 0:3] = d_pcld_cf
pcld_cf[:, 3] = pcld[:, 3]

d, _ = iu.camera_model.to_2d(d_pcld_cf, True)
g, _ = iu.camera_model.to_2d_dim(pcld_cf, 3, True)

d_pcld, v = iu.camera_model.to_3d(d, 0.0, 47.0)
g_pcld, _ = iu.camera_model.to_3d(g, -1.0, 100.0)
dpcld = d_pcld[v, :]
gpcld = g_pcld[v, 2]

pcld_valid = np.concatenate((dpcld, gpcld[:, np.newaxis]), axis=1)

bu = BenchmarkUtils(1, K)

# evaluate SOGMM
# model_path = '/Volumes/GoogleDrive/My Drive/phd/adaptive_perception/results/3dtk/model.pkl'
# with open(model_path, 'rb') as f:
#     loaded_model = pickle.load(f)

# sg = SOGMM()
# sg.model = loaded_model
# pr_pcd = sg.joint_dist_sample(pcld_valid.shape[0])
# print(calculate_depth_metrics(
#         np_to_o3d(pcld_valid), np_to_o3d(pr_pcd)))

# M = loaded_model.n_components_
# mem_bytes = 4 * M * (1 + 10 + 4)
# print(mem_bytes)

# regressed_path = '/Volumes/GoogleDrive/My Drive/phd/adaptive_perception/results/3dtk/regressed_thermobremen.txt'
# regressed_pcd = np.loadtxt(regressed_path)

# pr_g, _ = iu.camera_model.to_2d_dim(regressed_pcd, 3)


# N D Fucking T
l = LazyGrid(0.02)
n = NDTMap(l)
n.load_pointcloud(pcld_valid)
n.compute_ndt_cells_simple()

weights, means, covs = n.get_gaussians()
weights /= np.sum(weights)
n_components = np.shape(weights)[0]

model = GaussianMixture(n_components=n_components, covariance_type='full')
model.weights_ = weights
model.means_ = means
model.covariances_ = matrix_to_tensor(covs, 3)

pr_pcd, _ = model.sample(pcld_valid.shape[0])
print(calculate_depth_metrics(np_to_o3d(pcld_valid), np_to_o3d(pr_pcd)))

regressed_pcd = n.get_intensity_at_pcld(pcld_valid[:, 0:3])
pr_g, _ = iu.camera_model.to_2d_dim(regressed_pcd, 3, True)

g = np.nan_to_num(g)
pr_g = np.nan_to_num(pr_g)

fig, axs = plt.subplots(2, 1)

axs[0].imshow(g)
axs[1].imshow(pr_g)

print(calculate_color_metrics(g, pr_g))
plt.show()

M = model.n_components
mem_bytes = 4 * M * (1 + 3 + 6)
print(mem_bytes)