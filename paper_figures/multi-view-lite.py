import os
import numpy as np
import matplotlib.pyplot as plt

from sogmm_py.utils import *


results_path = '/Volumes/GoogleDrive/My Drive/phd/adaptive_perception/results'

# livingroom1
file_path = os.path.join(results_path, 'comps_livingroom1.npz')
stuff = np.load(file_path)
comp_data = stuff['arr_0']

copyroom_fig, copyroom_ax = plt.subplots()

copyroom_ax.plot(np.arange(0, len(comp_data)), comp_data)
plt.annotate("1", xy=(334, comp_data[334]+100),
             xycoords="data", bbox={"boxstyle": "circle", "color": "grey"})
plt.annotate("2", xy=(570, comp_data[570]-200),
             xycoords="data", bbox={"boxstyle": "circle", "color": "grey"})
plt.annotate("3", xy=(1814, comp_data[1814]+100),
             xycoords="data", bbox={"boxstyle": "circle", "color": "grey"})
plt.annotate("4", xy=(2340, comp_data[2340]-200),
             xycoords="data", bbox={"boxstyle": "circle", "color": "grey"})

plt.show()