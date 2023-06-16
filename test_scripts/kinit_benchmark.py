import numpy as np
import open3d as o3d
import time
from termcolor import cprint

from kinit_py import KInitf4CPU
from kinit_open3d_py import KInitf4GPU

from sogmm_py.utils import o3d_to_np

X = o3d_to_np(o3d.io.read_point_cloud('pcld_gt.pcd'))

N = X.shape[0]
K = 2000

resp_ref = np.zeros((N, K))
kinit_cpu = KInitf4CPU(K)
start = time.time()
dist_cpu = kinit_cpu.euclidean_dists(X[:K, :], X)
end = time.time()
cprint('time taken by kinit cpu dist %f seconds' % (end - start), 'yellow')

resp_ref = np.zeros((N, K))
kinit_gpu = KInitf4GPU(K)
start = time.time()
dist_gpu = kinit_gpu.euclidean_dists_gpu(X[:K, :], X)
end = time.time()
cprint('time taken by kinit gpu dist %f seconds' % (end - start), 'yellow')

np.testing.assert_array_almost_equal(dist_cpu, dist_gpu)

# resp_ref = np.zeros((N, K))
# kinit_cpu = KInitf4CPU(K)
# start = time.time()
# _, indices_cpu = kinit_cpu.resp_calc(X)
# end = time.time()
# cprint('time taken by kinit cpu %f seconds' % (end - start), 'yellow')

# resp_ref = np.zeros((N, K))
# kinit_gpu = KInitf4GPU(K)
# start = time.time()
# _, indices_gpu = kinit_gpu.resp_calc(X)
# end = time.time()
# cprint('time taken by kinit gpu %f seconds' % (end - start), 'yellow')

# np.testing.assert_array_almost_equal(indices_cpu, indices_gpu)