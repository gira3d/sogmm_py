import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import tikzplotlib as tz

def read_from_csv(file):
    ret = {}
    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        fn = csv_reader.fieldnames
        for row in csv_reader:
            ret[float(row[fn[0]])] = float(row[fn[1]])
    return ret

results_path = '/Volumes/GoogleDrive/My Drive/phd/adaptive_perception/results'
paper_fig_path = '/Users/kshitijgoel/Documents/main/papers.nosync/under_review/Paper-PRI-KG/data_analysis/figures/'
tikz_save = True

# quantitative analysis
psnr_fig, psnr_ax = plt.subplots()
err_fig, err_ax = plt.subplots()
mem_fig, mem_ax = plt.subplots()
xticklabels = []
width = 0.2
colors = ['crimson', 'navy', 'darkcyan']

# OctoMap
psnr_path_wall = os.path.join(results_path, 'stonewall_octomap/psnr.csv')
psnr_path_copier = os.path.join(results_path, 'copyroom_octomap/psnr.csv')
psnr_path_plant = os.path.join(results_path, 'lounge_octomap/psnr.csv')

wall_psnr_om = read_from_csv(psnr_path_wall)
copier_psnr_om = read_from_csv(psnr_path_copier)
plant_psnr_om = read_from_csv(psnr_path_plant)

idx = np.asarray([i for i in range(0, 2)])
psnr_ax.bar(idx - width, [val for _, val in sorted(wall_psnr_om.items(), reverse=True)], width=width, color=colors[0])
psnr_ax.bar(idx, [val for _, val in sorted(copier_psnr_om.items(), reverse=True)], width=width, color=colors[1])
psnr_ax.bar(idx + width, [val for _, val in sorted(plant_psnr_om.items(), reverse=True)], width=width, color=colors[2])

err_path_wall = os.path.join(results_path, 'stonewall_octomap/recon_mean.csv')
err_path_copier = os.path.join(results_path, 'copyroom_octomap/recon_mean.csv')
err_path_plant = os.path.join(results_path, 'lounge_octomap/recon_mean.csv')

wall_err_om = read_from_csv(err_path_wall)
copier_err_om = read_from_csv(err_path_copier)
plant_err_om = read_from_csv(err_path_plant)

idx = np.asarray([i for i in range(0, 2)])
err_ax.bar(idx - width, [val for _, val in sorted(wall_err_om.items(), reverse=True)], width=width, color=colors[0])
err_ax.bar(idx, [val for _, val in sorted(copier_err_om.items(), reverse=True)], width=width, color=colors[1])
err_ax.bar(idx + width, [val for _, val in sorted(plant_err_om.items(), reverse=True)], width=width, color=colors[2])

for ii, v in enumerate([val for _, val in sorted(wall_err_om.items(), reverse=True)]):
    err_ax.text(idx[ii] - (1.25 * width), v + 2e-4, f'{v:.4f}', color='blue', rotation=90)

for ii, v in enumerate([val for _, val in sorted(copier_err_om.items(), reverse=True)]):
    err_ax.text(idx[ii] - (0.25 * width), v + 2e-4, f'{v:.4f}', color='blue', rotation=90)

for ii, v in enumerate([val for _, val in sorted(plant_err_om.items(), reverse=True)]):
    err_ax.text(idx[ii] + (0.75 * width), v + 2e-4, f'{v:.4f}', color='blue', rotation=90)

mem_path_wall = os.path.join(results_path, 'stonewall_octomap/mem.csv')
mem_path_copier = os.path.join(results_path, 'copyroom_octomap/mem.csv')
mem_path_plant = os.path.join(results_path, 'lounge_octomap/mem.csv')

wall_mem_om = read_from_csv(mem_path_wall)
copier_mem_om = read_from_csv(mem_path_copier)
plant_mem_om = read_from_csv(mem_path_plant)

idx = np.asarray([i for i in range(0, 2)])
mem_ax.bar(idx - width, [val / 1e6 for _, val in sorted(wall_mem_om.items(), reverse=True)], width=width, color=colors[0])
mem_ax.bar(idx, [val / 1e6 for _, val in sorted(copier_mem_om.items(), reverse=True)], width=width, color=colors[1])
mem_ax.bar(idx + width, [val / 1e6 for _, val in sorted(plant_mem_om.items(), reverse=True)], width=width, color=colors[2])

xticklabels += ['OM ' + str(s) for s, _ in sorted(wall_psnr_om.items(), reverse=True)]

# NDTMap
psnr_path_wall = os.path.join(results_path, 'stonewall_ndtmap/psnr.csv')
psnr_path_copier = os.path.join(results_path, 'copyroom_ndtmap/psnr.csv')
psnr_path_plant = os.path.join(results_path, 'lounge_ndtmap/psnr.csv')

wall_psnr_ndt = read_from_csv(psnr_path_wall)
copier_psnr_ndt = read_from_csv(psnr_path_copier)
plant_psnr_ndt = read_from_csv(psnr_path_plant)

idx = np.asarray([i for i in range(2, 4)])
psnr_ax.bar(idx - width, [val for _, val in sorted(wall_psnr_ndt.items(), reverse=True)], width=width, color=colors[0])
psnr_ax.bar(idx, [val for _, val in sorted(copier_psnr_ndt.items(), reverse=True)], width=width, color=colors[1])
psnr_ax.bar(idx + width, [val for _, val in sorted(plant_psnr_ndt.items(), reverse=True)], width=width, color=colors[2])

err_path_wall = os.path.join(results_path, 'stonewall_ndtmap/recon_mean.csv')
err_path_copier = os.path.join(results_path, 'copyroom_ndtmap/recon_mean.csv')
err_path_plant = os.path.join(results_path, 'lounge_ndtmap/recon_mean.csv')

wall_err_ndt = read_from_csv(err_path_wall)
copier_err_ndt = read_from_csv(err_path_copier)
plant_err_ndt = read_from_csv(err_path_plant)

idx = np.asarray([i for i in range(2, 4)])
err_ax.bar(idx - width, [val for _, val in sorted(wall_err_ndt.items(), reverse=True)], width=width, color=colors[0])
err_ax.bar(idx, [val for _, val in sorted(copier_err_ndt.items(), reverse=True)], width=width, color=colors[1])
err_ax.bar(idx + width, [val for _, val in sorted(plant_err_ndt.items(), reverse=True)], width=width, color=colors[2])

for ii, v in enumerate([val for _, val in sorted(wall_err_ndt.items(), reverse=True)]):
    err_ax.text(idx[ii] - (1.25 * width), v + 2e-4, f'{v:.4f}', color='blue', rotation=90)

for ii, v in enumerate([val for _, val in sorted(copier_err_ndt.items(), reverse=True)]):
    err_ax.text(idx[ii] - (0.25 * width), v + 2e-4, f'{v:.4f}', color='blue', rotation=90)

for ii, v in enumerate([val for _, val in sorted(plant_err_ndt.items(), reverse=True)]):
    err_ax.text(idx[ii] + (0.75 * width), v + 2e-4, f'{v:.4f}', color='blue', rotation=90)

mem_path_wall = os.path.join(results_path, 'stonewall_ndtmap/mem.csv')
mem_path_copier = os.path.join(results_path, 'copyroom_ndtmap/mem.csv')
mem_path_plant = os.path.join(results_path, 'lounge_ndtmap/mem.csv')

wall_mem_ndt = read_from_csv(mem_path_wall)
copier_mem_ndt = read_from_csv(mem_path_copier)
plant_mem_ndt = read_from_csv(mem_path_plant)

idx = np.asarray([i for i in range(2, 4)])
mem_ax.bar(idx - width, [val / 1e6 for _, val in sorted(wall_mem_ndt.items(), reverse=True)], width=width, color=colors[0])
mem_ax.bar(idx, [val / 1e6 for _, val in sorted(copier_mem_ndt.items(), reverse=True)], width=width, color=colors[1])
mem_ax.bar(idx + width, [val / 1e6 for _, val in sorted(plant_mem_ndt.items(), reverse=True)], width=width, color=colors[2])

xticklabels += ['NDT ' + str(s) for s, _ in sorted(wall_psnr_ndt.items(), reverse=True)]

# FC
psnr_path_wall = os.path.join(results_path, 'stonewall_fixed_comps/psnr.csv')
psnr_path_copier = os.path.join(results_path, 'copyroom_fixed_comps/psnr.csv')
psnr_path_plant = os.path.join(results_path, 'lounge_fixed_comps/psnr.csv')

wall_psnr_fc = read_from_csv(psnr_path_wall)
copier_psnr_fc = read_from_csv(psnr_path_copier)
plant_psnr_fc = read_from_csv(psnr_path_plant)

idx = np.asarray([i for i in range(4, 7)])
psnr_ax.bar(idx - width, [val for _, val in sorted(wall_psnr_fc.items())], width=width, color=colors[0])
psnr_ax.bar(idx, [val for _, val in sorted(copier_psnr_fc.items())], width=width, color=colors[1])
psnr_ax.bar(idx + width, [val for _, val in sorted(plant_psnr_fc.items())], width=width, color=colors[2])

err_path_wall = os.path.join(results_path, 'stonewall_fixed_comps/recon_mean.csv')
err_path_copier = os.path.join(results_path, 'copyroom_fixed_comps/recon_mean.csv')
err_path_plant = os.path.join(results_path, 'lounge_fixed_comps/recon_mean.csv')

wall_err_fc = read_from_csv(err_path_wall)
copier_err_fc = read_from_csv(err_path_copier)
plant_err_fc = read_from_csv(err_path_plant)

idx = np.asarray([i for i in range(4, 7)])
err_ax.bar(idx - width, [val for _, val in sorted(wall_err_fc.items())], width=width, color=colors[0])
err_ax.bar(idx, [val for _, val in sorted(copier_err_fc.items())], width=width, color=colors[1])
err_ax.bar(idx + width, [val for _, val in sorted(plant_err_fc.items())], width=width, color=colors[2])

for ii, v in enumerate([val for _, val in sorted(wall_err_fc.items())]):
    err_ax.text(idx[ii] - (1.25 * width), v + 2e-4, f'{v:.4f}', color='blue', rotation=90)

for ii, v in enumerate([val for _, val in sorted(copier_err_fc.items())]):
    err_ax.text(idx[ii] - (0.25 * width), v + 2e-4, f'{v:.4f}', color='blue', rotation=90)

for ii, v in enumerate([val for _, val in sorted(plant_err_fc.items())]):
    err_ax.text(idx[ii] + (0.75 * width), v + 2e-4, f'{v:.4f}', color='blue', rotation=90)

mem_path_wall = os.path.join(results_path, 'stonewall_fixed_comps/mem.csv')
mem_path_copier = os.path.join(results_path, 'copyroom_fixed_comps/mem.csv')
mem_path_plant = os.path.join(results_path, 'lounge_fixed_comps/mem.csv')

wall_mem_fc = read_from_csv(mem_path_wall)
copier_mem_fc = read_from_csv(mem_path_copier)
plant_mem_fc = read_from_csv(mem_path_plant)

idx = np.asarray([i for i in range(4, 7)])
mem_ax.bar(idx - width, [val / 1e6 for _, val in sorted(wall_mem_fc.items())], width=width, color=colors[0])
mem_ax.bar(idx, [val / 1e6 for _, val in sorted(copier_mem_fc.items())], width=width, color=colors[1])
mem_ax.bar(idx + width, [val / 1e6 for _, val in sorted(plant_mem_fc.items())], width=width, color=colors[2])

xticklabels += ['FC ' + str(int(s)) for s, _ in sorted(wall_psnr_fc.items())]

# SOGMM
psnr_path_wall = os.path.join(results_path, 'stonewall_sogmm/psnr.csv')
psnr_path_copier = os.path.join(results_path, 'copyroom_sogmm/psnr.csv')
psnr_path_plant = os.path.join(results_path, 'lounge_sogmm/psnr.csv')

wall_psnr_sogmm = read_from_csv(psnr_path_wall)
copier_psnr_sogmm = read_from_csv(psnr_path_copier)
plant_psnr_sogmm = read_from_csv(psnr_path_plant)

idx = np.asarray([i for i in range(7, 10)])
psnr_ax.bar(idx - width, [val for _, val in sorted(wall_psnr_sogmm.items(), reverse=True)], width=width, color=colors[0])
psnr_ax.bar(idx, [val for _, val in sorted(copier_psnr_sogmm.items(), reverse=True)], width=width, color=colors[1])
psnr_ax.bar(idx + width, [val for _, val in sorted(plant_psnr_sogmm.items(), reverse=True)], width=width, color=colors[2])

err_path_wall = os.path.join(results_path, 'stonewall_sogmm/recon_mean.csv')
err_path_copier = os.path.join(results_path, 'copyroom_sogmm/recon_mean.csv')
err_path_plant = os.path.join(results_path, 'lounge_sogmm/recon_mean.csv')

wall_err_sogmm = read_from_csv(err_path_wall)
copier_err_sogmm = read_from_csv(err_path_copier)
plant_err_sogmm = read_from_csv(err_path_plant)

idx = np.asarray([i for i in range(7, 10)])
err_ax.bar(idx - width, [val for _, val in sorted(wall_err_sogmm.items(), reverse=True)], width=width, color=colors[0])
err_ax.bar(idx, [val for _, val in sorted(copier_err_sogmm.items(), reverse=True)], width=width, color=colors[1])
err_ax.bar(idx + width, [val for _, val in sorted(plant_err_sogmm.items(), reverse=True)], width=width, color=colors[2])

for ii, v in enumerate([val for _, val in sorted(wall_err_sogmm.items(), reverse=True)]):
    err_ax.text(idx[ii] - (1.25 * width), v + 2e-4, f'{v:.4f}', color='blue', rotation=90)

for ii, v in enumerate([val for _, val in sorted(copier_err_sogmm.items(), reverse=True)]):
    err_ax.text(idx[ii] - (0.25 * width), v + 2e-4, f'{v:.4f}', color='blue', rotation=90)

for ii, v in enumerate([val for _, val in sorted(plant_err_sogmm.items(), reverse=True)]):
    err_ax.text(idx[ii] + (0.75 * width), v + 2e-4, f'{v:.4f}', color='blue', rotation=90)

mem_path_wall = os.path.join(results_path, 'stonewall_sogmm/mem.csv')
mem_path_copier = os.path.join(results_path, 'copyroom_sogmm/mem.csv')
mem_path_plant = os.path.join(results_path, 'lounge_sogmm/mem.csv')

wall_mem_sogmm = read_from_csv(mem_path_wall)
copier_mem_sogmm = read_from_csv(mem_path_copier)
plant_mem_sogmm = read_from_csv(mem_path_plant)

idx = np.asarray([i for i in range(7, 10)])
mem_ax.bar(idx - width, [val / 1e6 for _, val in sorted(wall_mem_sogmm.items(), reverse=True)], width=width, color=colors[0])
mem_ax.bar(idx, [val / 1e6 for _, val in sorted(copier_mem_sogmm.items(), reverse=True)], width=width, color=colors[1])
mem_ax.bar(idx + width, [val / 1e6 for _, val in sorted(plant_mem_sogmm.items(), reverse=True)], width=width, color=colors[2])

xticklabels += ['SOGMM ' + str(s) for s, _ in sorted(wall_psnr_sogmm.items(), reverse=True)]

# wrap-up this figure
psnr_ax.set_xticks(np.asarray([i for i in range(0, 10)]))
psnr_ax.set_xticklabels(xticklabels, rotation=15)
# psnr_ax.legend(['Wall', 'Copier', 'Plant'])
psnr_ax.set_ylabel('PSNR')

if tikz_save:
    tz.save(figure=psnr_fig, filepath=os.path.join(paper_fig_path, 'single-view-psnr.tex'))

err_ax.set_xticks(np.asarray([i for i in range(0, 10)]))
err_ax.set_xticklabels(xticklabels, rotation=15)
# err_ax.legend(['Wall', 'Copier', 'Plant'])
err_ax.set_ylabel('Recon. Err. (m)')

if tikz_save:
    tz.save(figure=err_fig, filepath=os.path.join(paper_fig_path, 'single-view-err.tex'))

mem_ax.set_xticks(np.asarray([i for i in range(0, 10)]))
mem_ax.set_xticklabels(xticklabels, rotation=15)
# mem_ax.legend(['Wall', 'Copier', 'Plant'])
mem_ax.set_ylabel('Memory (MB)')

if tikz_save:
    tz.save(figure=mem_fig, filepath=os.path.join(paper_fig_path, 'single-view-mem.tex'))

plt.show()