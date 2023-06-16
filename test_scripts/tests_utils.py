import os
import numpy as np
import matplotlib as mpl


def make_ellipse(mean, covariance, color, ax):
    v, w = np.linalg.eigh(covariance)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180.0 * angle / np.pi  # convert to degrees
    v = 3.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(
        mean, v[0], v[1], 180 + angle, color=color
    )
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.4)
    ax.add_artist(ell)
    ax.set_aspect("equal")


def make_ellipses_gmm(gmm, ax, case='py', color='red'):
    for n, mean in enumerate(gmm.means_):
        if case == 'py':
            covariances = gmm.covariances_[n][:2, :2]
            make_ellipse(mean, covariances, color, ax)
        else:
            covariances = matrix_to_tensor(gmm.covariances_)[n][:2, :2]
            make_ellipse(mean, covariances, color, ax)


def tensor_to_matrix(tensor):
    n_components, _, _ = tensor.shape
    matrix = np.zeros((n_components, 4))
    for i in range(n_components):
        matrix[i, :] = tensor[i, :, :].flatten(order='C')

    return matrix


def matrix_to_tensor(matrix):
    n_components, _ = matrix.shape
    tensor = np.zeros((n_components, 2, 2))
    for i in range(n_components):
        tensor[i, :, :] = matrix[i, :].reshape((2, 2))

    return tensor


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def make_dir(str1, str2):
    path = os.path.join(str1, str2)

    if not os.path.exists(path):
        os.makedirs(path)
        return path
    else:
        return path
