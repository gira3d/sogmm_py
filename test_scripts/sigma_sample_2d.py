import numpy as np

# Datasets
from sklearn import datasets

# GMR Python
from gmr import GMM, plot_error_ellipses

# Plotting
import matplotlib.pyplot as plt

K = 2
data, _ = datasets.make_blobs(n_samples=1000, centers=K, random_state=11)

gmm = GMM(n_components=K, random_state=1)
gmm.from_samples(data)

fig, ax = plt.subplots()

ax.scatter(data[:, 0], data[:, 1], s=1)
plot_error_ellipses(ax, gmm, factors=np.linspace(1.0, 3.0, 3))

pts = []
for i in range(K):
    cov = gmm.covariances[i]
    mean = gmm.means[i]
    eigval, eigvec = np.linalg.eigh(cov)

    sorted_idxs = np.argsort(eigval)

    l = np.sqrt(eigval[sorted_idxs])

    sigmas = [1.0, 2.0, 3.0]

    for sigma in sigmas:
        u1 = eigvec[:, sorted_idxs[0]]
        v = sigma * l[0] * u1
        pts.append(np.array([mean[0] + v[0], mean[1] + v[1]]))
        pts.append(np.array([mean[0] - v[0], mean[1] - v[1]]))

        u2 = eigvec[:, sorted_idxs[1]]
        v = sigma * l[1] * u2
        pts.append(np.array([mean[0] + v[0], mean[1] + v[1]]))
        pts.append(np.array([mean[0] - v[0], mean[1] - v[1]]))

        u12 = (1.0 / np.sqrt(2.0)) * (eigvec[:, sorted_idxs[0]] + eigvec[:, sorted_idxs[1]])
        v = sigma * l[1] * u12
        pts.append(np.array([mean[0] + v[0], mean[1] + v[1]]))
        pts.append(np.array([mean[0] - v[0], mean[1] - v[1]]))

        u21 = (1.0 / np.sqrt(2.0)) * (-eigvec[:, sorted_idxs[0]] + eigvec[:, sorted_idxs[1]])
        v = sigma * l[0] * u21
        pts.append(np.array([mean[0] + v[0], mean[1] + v[1]]))
        pts.append(np.array([mean[0] - v[0], mean[1] - v[1]]))

        pts.append(np.array([mean[0], mean[1]]))
    
    ax.arrow(mean[0], mean[1], u1[0], u1[1], color='k', width=0.05, length_includes_head=True)
    ax.arrow(mean[0], mean[1], -u1[0], -u1[1], color='k', linestyle='--', width=0.05, length_includes_head=True)
    ax.arrow(mean[0], mean[1], u2[0], u2[1], color='k', width=0.05, length_includes_head=True)
    ax.arrow(mean[0], mean[1], -u2[0], -u2[1], color='k', linestyle='--', width=0.05, length_includes_head=True)
    ax.arrow(mean[0], mean[1], u12[0], u12[1], color='green', width=0.05, length_includes_head=True)
    ax.arrow(mean[0], mean[1], -u12[0], -u12[1], color='green', linestyle='--', width=0.05, length_includes_head=True)
    ax.arrow(mean[0], mean[1], u21[0], u21[1], color='green', width=0.05, length_includes_head=True)
    ax.arrow(mean[0], mean[1], -u21[0], -u21[1], color='green', linestyle='--', width=0.05, length_includes_head=True)

pts = np.array(pts)
ax.scatter(pts[:, 0], pts[:, 1], s=20, c='red')

plt.show()