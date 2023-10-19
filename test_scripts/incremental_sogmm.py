import argparse
import os
import numpy as np
import glob
from scipy.spatial import KDTree
from nigh_py import R3KDTree
from nigh_py import SO3KDTree
from nigh_py import SE3KDTree

from sogmm_py.sogmm import SOGMM
from sogmm_py.utils import *
from sogmm_py.vis_open3d import VisOpen3D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Incremental-SOGMM")
    parser.add_argument('--datasetroot', type=str)

    args = parser.parse_args()

    dfolder = args.datasetroot
    dname = 'livingroom1'

    # paths to all rgb and depth images in the dataset
    rgb_paths = sorted(glob.glob(os.path.join(dfolder, dname + '-color/*.jpg')))
    depth_paths = sorted(glob.glob(os.path.join(dfolder, dname + '-depth/*.png')))

    # read the full camera trajectory
    traj = read_log_trajectory(os.path.join(dfolder, dname + '-traj.log'))

    # decimation -- mostly to test rapidly
    deci = 4.0
    K_d = np.eye(3)
    K_d[0, 0] = 525.0/deci
    K_d[1, 1] = 525.0/deci
    K_d[0, 2] = 319.5/deci
    K_d[1, 2] = 239.5/deci
    W_d = (int)(640/deci)
    H_d = (int)(480/deci)
    iu_d = ImageUtils(K_d, im_h=H_d, im_w=W_d)

    # SOGMM
    bandwidth = 0.015
    sg = SOGMM(bandwidth, compute='GPU')

    # frame to learn model from
    frame_1 = 765
    pose_1 = traj[frame_1].pose
    translation_1 = pose_1[:3, 3]
    rotation_1 = pose_1[:3, :3].flatten()
    pcld_1, im_1 = iu_d.generate_pcld_wf(pose_1, rgb_path=rgb_paths[frame_1],
                                        depth_path=depth_paths[frame_1], size=(W_d, H_d))
    sg.fit(pcld_1)

    # second frame
    frame_2 = 800
    pose_2 = traj[frame_2].pose
    translation_2 = pose_2[:3, 3]
    pcld_2, im_2 = iu_d.generate_pcld_wf(pose_2, rgb_path=rgb_paths[frame_2],
                                        depth_path=depth_paths[frame_2], size=(W_d, H_d))

    # third frame
    frame_3 = 100
    pose_3 = traj[frame_3].pose
    translation_3 = pose_3[:3, 3]
    rotation_3 = pose_3[:3, :3].flatten()
    pcld_3, im_3 = iu_d.generate_pcld_wf(pose_3, rgb_path=rgb_paths[frame_3],
                                        depth_path=depth_paths[frame_3], size=(W_d, H_d))
    sg.fit(pcld_3)

    # Visualizer
    vis = VisOpen3D(visible=True)

    # create a KDTree of translations
    translations = np.zeros((2, 3))
    translations[0, :] = translation_1
    translations[1, :] = translation_3
    # tree = R3KDTree(3.0)
    # tree = SO3KDTree(np.pi / 4.0)
    tree = SE3KDTree(2.0 + np.pi / 4.0)

    # tree.insert(translations)

    # rotations = np.zeros((2, 9))
    # rotations[0, :] = rotation_1
    # rotations[1, :] = rotation_3
    # tree.insert(rotations)

    rotations = np.zeros((2, 9))
    rotations[0, :] = rotation_1
    rotations[1, :] = rotation_3
    tree.insert(translations, rotations)

    # out = tree.search(translation_2)

    # rotation_2_mat = pose_2[:3, :3]
    # out = tree.search(rotation_2_mat)

    rotation_2_mat = pose_2[:3, :3]
    out = tree.search(translation_2, rotation_2_mat)

    scores = None
    resampled_pclds = list()
    for o in out:
        resampled_pcld = sg.local_models[int(o[0].name)].sample(640 * 480, 2.2)
        resampled_pclds.append(resampled_pcld)
        if scores is None:
            scores = sg.local_models[int(o[0].name)].score_samples(pcld_2)
        else:
            scores = np.maximum(scores, sg.local_models[int(o[0].name)].score_samples(pcld_2))

    scores = scores.flatten()
    novel_pts = pcld_2[scores < -10, :]
    vis.add_geometry(np_to_o3d(novel_pts[:, :3], color=[1.0, 0.0, 0.0]))

    vis.frustrum(pose_1, K_d, W_d, H_d, color=[1.0, 0.0, 0.0])
    vis.frustrum(pose_2, K_d, W_d, H_d, color=[0.0, 1.0, 0.0])
    vis.frustrum(pose_3, K_d, W_d, H_d, color=[0.0, 0.0, 1.0])

    for r in resampled_pclds:
        vis.add_geometry(np_to_o3d(r))

    vis.render()