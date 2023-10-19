import numpy as np

class GMMSpatialHash:
    def __init__(self, resolution, width=100, height=100, depth=100):
        self.w = (int)(width)
        self.h = (int)(height)
        self.d = (int)(depth)
        self.res = resolution
        self.o = (-1.0 * resolution / 2.0) * np.array([width, height, depth])

        self.hash_table = {}

    def add_point(self, key, value):
        idx = self.point_to_index(key)

        try:
            temp = self.hash_table[idx]
        except KeyError:
            self.hash_table[idx] = list()

        self.hash_table[idx].append(value)

    def add_points(self, points, values):
        N = points.shape[0]
        for i in range(N):
            self.add_point(points[i, :3], values[i])

    def point_to_index(self, point):
        r = (int)((point[1] - self.o[1]) / self.res)
        c = (int)((point[0] - self.o[0]) / self.res)
        s = (int)((point[2] - self.o[2]) / self.res)

        return (r*self.w + c)*self.d + s

    def find_point(self, key):
        idx = self.point_to_index(key)

        try:
            temp = self.hash_table[idx]
            return temp
        except KeyError:
            return [-1]

    def find_points(self, points):
        N = points.shape[0]
        results = []
        for i in range(N):
            results += self.find_point(points[i, :3])

        results = np.array(results, dtype=int)

        return np.unique(results[results >= 0])

    def find_buckets(self, points):
        N = points.shape[0]
        bucket_ids = []
        buckets = []
        for i in range(N):
            bucket_id = self.point_to_index(points[i, :3])
            try:
                loc = bucket_ids.index(bucket_id)
                buckets[loc].append(i)
            except ValueError:
                bucket_ids.append(bucket_id)
                buckets.append([i])

        assert(len(bucket_ids) == len(buckets))

        return bucket_ids, buckets

    def process_idx(self, idx):
        try:
            return np.unique(np.array(self.hash_table[idx]))
        except KeyError:
            return None

    def add_sigma_points(self, means, covs, values):
        K = means.shape[0]
        for i in range(K):
            mean = means[i]
            cov = covs[i]
            eigval, eigvec = np.linalg.eigh(cov)

            sorted_idxs = np.argsort(eigval)[1:3]

            l = np.sqrt(eigval[sorted_idxs])

            sigmas = [1.0, 2.0, 3.0]

            for sigma in sigmas:
                u1 = eigvec[:, sorted_idxs[0]]
                v = sigma * l[0] * u1
                self.add_point(np.array([mean[0] + v[0], mean[1] + v[1], mean[2]]), values[i])
                self.add_point(np.array([mean[0] - v[0], mean[1] - v[1], mean[2]]), values[i])

                u2 = eigvec[:, sorted_idxs[1]]
                v = sigma * l[1] * u2
                self.add_point(np.array([mean[0] + v[0], mean[1] + v[1], mean[2]]), values[i])
                self.add_point(np.array([mean[0] - v[0], mean[1] - v[1], mean[2]]), values[i])

                u12 = (1.0 / np.sqrt(2.0)) * (eigvec[:, sorted_idxs[0]] + eigvec[:, sorted_idxs[1]])
                v = sigma * l[1] * u12
                self.add_point(np.array([mean[0] + v[0], mean[1] + v[1], mean[2]]), values[i])
                self.add_point(np.array([mean[0] - v[0], mean[1] - v[1], mean[2]]), values[i])

                u21 = (1.0 / np.sqrt(2.0)) * (-eigvec[:, sorted_idxs[0]] + eigvec[:, sorted_idxs[1]])
                v = sigma * l[0] * u21
                self.add_point(np.array([mean[0] + v[0], mean[1] + v[1], mean[2]]), values[i])
                self.add_point(np.array([mean[0] - v[0], mean[1] - v[1], mean[2]]), values[i])

                self.add_point(means[i, :3], values[i])