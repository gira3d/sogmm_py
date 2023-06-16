import os
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import open3d as o3d

filepath = "/Users/wtabib/Downloads/crevasse_cleaned.pcd"
pcd_data = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud(filepath)
#o3d.visualization.draw_geometries([pcd])
points = np.asarray(pcd.points)
points = points[~np.all(points == 0, axis=1)]  # removes all zeros

num_iter = 20
num_components = 100

gmm_standard_scores = np.zeros((num_iter, 1))
gmm_kmeanspp_scores = np.zeros((num_iter, 1))

for i in range(0, num_iter):
    print('iteration: ' + str(i))

    gmm = GaussianMixture(num_components)
    gmm = gmm.fit(points)
    gmm_standard_scores[i, 0] = gmm.score(points)
    print('score from regular fit: ' + str(gmm_standard_scores[i, 0]))

    gmm = GaussianMixture(num_components, init_params="k-means++")
    gmm = gmm.fit(points)
    gmm_kmeanspp_scores[i, 0] = gmm.score(points)
    print('score from kmeans++ fit: ' + str(gmm_kmeanspp_scores[i, 0]))

print('mean from regular fit:  ' + str(np.mean(gmm_standard_scores)))
print('std from regular fit:   ' + str(np.std(gmm_standard_scores)))
print('mean from kmeanspp fit: ' + str(np.mean(gmm_kmeanspp_scores)))
print('std from kmeanspp fit:  ' + str(np.std(gmm_kmeanspp_scores)))
