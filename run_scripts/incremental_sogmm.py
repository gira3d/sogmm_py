import os
import open3d as o3d
import glob
import numpy as np
import matplotlib.pyplot as plt

from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import ImageUtils, read_log_trajectory, np_to_o3d, calculate_color_metrics

dt_pth = '/home/kshitijgoel/Documents/datasets'

dataset = 'livingroom1'  # sofa is 796 to 1066
dpth_ext = '.png'
rgb_ext = '.jpg'

decimate_factor = 5
bandwidth = 0.01
sg = SOGMM(bandwidth, compute='GPU')

K = np.eye(3)
K[0, 0] = 525.0
K[1, 1] = 525.0
K[0, 2] = 319.5
K[1, 2] = 239.5
iu = ImageUtils(K)
W = 640
H = 480

K_d = np.eye(3)
K_d[0, 0] = 525.0/decimate_factor
K_d[1, 1] = 525.0/decimate_factor
K_d[0, 2] = 319.5/decimate_factor
K_d[1, 2] = 239.5/decimate_factor
iu_d = ImageUtils(K_d)
W_d = (int)(640/decimate_factor)
H_d = (int)(480/decimate_factor)

depth_paths = sorted(glob.glob(os.path.join(dt_pth, 
                                            'livingroom1-depth/*' + dpth_ext)))
rgb_paths = sorted(glob.glob(os.path.join(dt_pth,
                                          'livingroom1-color/*' + rgb_ext)))
traj = read_log_trajectory(os.path.join(dt_pth, dataset + '-traj.log'))

frame_1 = 920
frame_2 = 921
pose_1 = traj[frame_1].pose
pose_2 = traj[frame_2].pose

# ground truth points for frame 1 and frame 2
gt_pcld_1, gt_im_1 = iu.generate_pcld_wf(pose_1, rgb_path=rgb_paths[frame_1],
                                         depth_path=depth_paths[frame_1], size=(W, H))

# learn model for frame 1 using the decimated point cloud
pcld_1, im_1 = iu_d.generate_pcld_wf(pose_1, rgb_path=rgb_paths[frame_1],
                                     depth_path=depth_paths[frame_1], size=(W_d, H_d))
model_1 = sg.fit(pcld_1)

print('model fidelity (num clusters)', sg.model.n_components_)

# sg.pickle_global_model(path='../../sogmm_open3d/test/model.pkl')
# o3d.io.write_point_cloud('../../sogmm_open3d/test/test_color_conditional.pcd', np_to_o3d(gt_pcld_1))
# np.savetxt('../../sogmm_open3d/test/test_color_conditional_pose.txt', pose_1)

# psnr before frame 2 is incorporated
regressed_colors = np.zeros((gt_pcld_1.shape[0], 1))
_, regressed_colors, regressed_variance = sg.model.color_conditional(gt_pcld_1[:, 0:3])

regressed_color_pcd = np.zeros(gt_pcld_1.shape)
regressed_color_pcd[:, 0:3] = gt_pcld_1[:, 0:3]
regressed_color_pcd[:, 3] = np.squeeze(regressed_colors)

regressed_uncert_pcd = np.zeros(gt_pcld_1.shape)
regressed_uncert_pcd[:, 0:3] = gt_pcld_1[:, 0:3]
regressed_uncert_pcd[:, 3] = np.squeeze(regressed_variance)

_, pr_g = iu.pcld_wf_to_imgs(pose_1, regressed_color_pcd)
_, pr_u_g = iu.pcld_wf_to_imgs(pose_1, regressed_uncert_pcd)

# check if the predicted image looks good or not
fig, ax = plt.subplots(3, layout='compressed')
ax[0].imshow(np.asarray(gt_im_1.color))
ax[1].imshow(pr_g)
pr_u_mappable = ax[2].imshow(pr_u_g)

fig.colorbar(pr_u_mappable, ax=ax[2])

plt.show()

# psnr, ssim = calculate_color_metrics(np.asarray(gt_im_1.color), pr_g)
# print('psnr before incorporating frame 2: %f' % psnr)

# reconstruction from model_1 as a sanity check
# pts_1 = model_1.sample(100000, 2.5)
# o3d.visualization.draw_geometries([np_to_o3d(pts_1)])

# get points for frame 2 and score them with model_1
# likelihood_thres = -10
# gt_pcld_2, gt_im_2 = iu.generate_pcld_wf(pose_2, rgb_path=rgb_paths[frame_2],
#                                          depth_path=depth_paths[frame_2], size=(W, H))
# pcld_2, im_2 = iu_d.generate_pcld_wf(pose_2, rgb_path=rgb_paths[frame_2],
#                                      depth_path=depth_paths[frame_2], size=(W_d, H_d))
# scores_2 = model_1.score_samples(pcld_2)

# overlap_ids = np.where(scores_2 >= likelihood_thres)[0]
# overlap_pts = pcld_2[overlap_ids, :]

# o3d.visualization.draw_geometries([np_to_o3d(overlap_pts)])