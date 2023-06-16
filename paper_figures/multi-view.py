import os
import numpy as np
import open3d as o3d
from termcolor import cprint
import pickle
from sogmm_py.vis_open3d import VisOpen3D
import glob
import matplotlib.pyplot as plt

from sogmm_py.utils import *

# used in open3d visualization
O3D_K = np.array([[935.30743609,   0.,         959.5],
                    [0.,         935.30743609, 539.5],
                    [0.,           0.,           1.]])

results_path = '/Volumes/GoogleDrive/My Drive/phd/adaptive_perception/results'

# quantitative analysis
# wall
local_model_paths = glob.glob(os.path.join(results_path, 'stonewall_sogmm/stonewall_0.01/multi-view/local_model_*.pkl'))
keyframes = sorted([int(os.path.splitext(l)[0].split('/')[-1].split('_')[-1]) for l in local_model_paths])

C = []
prev_count = 0
lowest_count = np.inf
lowest_count_key = 0
highest_count = 0
lowest_count_key = 0
for k in keyframes:
    mp = os.path.join(results_path, 'stonewall_sogmm/stonewall_0.01/multi-view/local_model_' + str(k) + '.pkl')
    with open(mp, 'rb') as f:
        model = pickle.load(f)
    count = model.n_components_ - prev_count
    if count > highest_count:
        highest_count = count
        highest_count_key = k
    if count < lowest_count:
        lowest_count = count
        lowest_count_key = k
    C.append(model.n_components_ - prev_count)
    prev_count = model.n_components_

fmp = os.path.join(results_path, 'stonewall_sogmm/stonewall_0.01/multi-view/full_model.pkl')
with open(fmp, 'rb') as f:
    full_model = pickle.load(f)

fig, ax = plt.subplots()
ax.plot(keyframes, C, 'o-')
print(lowest_count_key, lowest_count)
print(highest_count_key, highest_count)

# copyroom
local_model_paths = glob.glob(os.path.join(results_path, 'copyroom_sogmm/copyroom_0.01/multi-view/local_model_*.pkl'))
keyframes = sorted([int(os.path.splitext(l)[0].split('/')[-1].split('_')[-1]) for l in local_model_paths])

C = []
prev_count = 0
lowest_count = np.inf
lowest_count_key = 0
highest_count = 0
lowest_count_key = 0
for k in keyframes:
    mp = os.path.join(results_path, 'copyroom_sogmm/copyroom_0.01/multi-view/local_model_' + str(k) + '.pkl')
    with open(mp, 'rb') as f:
        model = pickle.load(f)
    count = model.n_components_ - prev_count
    if count > highest_count:
        highest_count = count
        highest_count_key = k
    if count < lowest_count:
        lowest_count = count
        lowest_count_key = k
    C.append(model.n_components_ - prev_count)
    prev_count = model.n_components_

fmp = os.path.join(results_path, 'copyroom_sogmm/copyroom_0.01/multi-view/full_model.pkl')
with open(fmp, 'rb') as f:
    full_model = pickle.load(f)

fig, ax = plt.subplots()
ax.plot(keyframes, C, 'o-')
print(lowest_count_key, lowest_count)
print(highest_count_key, highest_count)

# lounge
local_model_paths = glob.glob(os.path.join(results_path, 'lounge_sogmm/lounge_0.01/multi-view/local_model_*.pkl'))
keyframes = sorted([int(os.path.splitext(l)[0].split('/')[-1].split('_')[-1]) for l in local_model_paths])

C = []
prev_count = 0
lowest_count = np.inf
lowest_count_key = 0
highest_count = 0
lowest_count_key = 0
for k in keyframes:
    mp = os.path.join(results_path, 'lounge_sogmm/lounge_0.01/multi-view/local_model_' + str(k) + '.pkl')
    with open(mp, 'rb') as f:
        model = pickle.load(f)
    count = model.n_components_ - prev_count
    if count > highest_count:
        highest_count = count
        highest_count_key = k
    if count < lowest_count:
        lowest_count = count
        lowest_count_key = k
    C.append(model.n_components_ - prev_count)
    prev_count = model.n_components_

fmp = os.path.join(results_path, 'lounge_sogmm/lounge_0.01/multi-view/full_model.pkl')
with open(fmp, 'rb') as f:
    full_model = pickle.load(f)

fig, ax = plt.subplots()
ax.plot(keyframes, C, 'o-')
print(lowest_count_key, lowest_count)
print(highest_count_key, highest_count)

plt.show()