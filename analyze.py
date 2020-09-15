import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm
from scipy import stats
import os
import pickle
from utils import ROIs, count_violation_pairs, decode_data, cal_min_dists_all_frame, custom_simple_linear_regression


dict_dataset_names = {
    'oxford_town': 'Oxford Town Center Dataset',
    'mall': 'Mall Dataset',
    'grand_central': 'Train Station Dataset'
}

path_results = 'results/test_faster_rcnn'  # faster RCNN results
# path_results = 'additional_detectors/yolo_v4_darknet_official/results/result_yolo_v4_darknet_official'


def analyze_statistics(dataset):
    print('=======================')
    print('Processing %s ...' % dataset)

    path_result = os.path.join(path_results, dataset)
    path_analysis = os.path.join(path_results, 'analysis')

    os.makedirs(path_result, exist_ok=True)
    os.makedirs(path_analysis, exist_ok=True)

    data = pickle.load(open(os.path.join(path_result, 'statistic_data.p'), 'rb'))
    roi = ROIs[dataset]
    x_min, x_max, y_min, y_max = roi
    area = (x_max - x_min) * (y_max - y_min)
    indexs_frame, ts_inference, pts_roi_all_frame, density, nums_ped = decode_data(data=data, roi=roi)
    print('Mean inference time = %.6f' % np.mean(ts_inference))

    all_min_dists, min_min_dists, avg_min_dists = cal_min_dists_all_frame(pts_roi_all_frame)
    violations = count_violation_pairs(pts_all_frames=pts_roi_all_frame)

    none_indexes = np.where(avg_min_dists == None)[0]
    indexs_frame = np.delete(indexs_frame, none_indexes, 0)
    density = np.delete(density, none_indexes, 0)
    min_min_dists = np.delete(min_min_dists, none_indexes, 0)
    avg_min_dists = np.delete(avg_min_dists, none_indexes, 0)
    nums_ped = np.delete(nums_ped, none_indexes, 0)
    violations = np.delete(violations, none_indexes, 0)
    # violations = violations / nums_ped
    # violations, density = density, violations


    # figure 1 - min dists
    fig = plt.figure(figsize=(5., 3.))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.15)
    fig.suptitle(dict_dataset_names[dataset])

    ax = fig.add_subplot(111)
    ax.hist(all_min_dists, bins=100)
    ax.grid()
    ax.set_xlabel(r'Closest Distance ($m$)')
    ax.set_ylabel('Count')
    ax.set(xlim=(0, 10))

    fig.savefig(os.path.join(path_analysis, dataset + '_closest_dists.png'))
    plt.close(fig)

    # figure 2 - min-min dists
    fig = plt.figure(figsize=(5., 3.))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.15)
    fig.suptitle(dict_dataset_names[dataset])

    ax = fig.add_subplot(111)
    ax.hist(min_min_dists, bins=100)
    ax.grid()
    ax.set_xlabel(r'Minimal Closest Distance ($m$)')
    ax.set_ylabel('Count')

    fig.savefig(os.path.join(path_analysis, dataset + '_min_closest_dists.png'))
    plt.close(fig)

    # figure 3 - density, min dist, min_min dist over time
    t_max = 300

    if dataset == 'oxford_town':
        ts = indexs_frame / 10
    elif dataset == 'mall':
        ts = indexs_frame / 1
    elif dataset == 'grand_central':
        ts = indexs_frame / 25
    else:
        raise Exception('invalid dataset')
    extracted = ts <= t_max

    fig = plt.figure(figsize=(11.47, 4.33))
    fig.subplots_adjust(left=0.07, bottom=0.12, right=0.98, top=0.91, hspace=0.36)
    fig.suptitle(dict_dataset_names[dataset])

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(ts[extracted], avg_min_dists[extracted], '.-', label=r'avg. closest physical distance $d_{avg}$ ($m$)')
    ax.plot(ts[extracted], min_min_dists[extracted], '.-', label=r'min. closest physical distance $d_{min}$ ($m$)')
    # ax.plot(ts[extracted], violations[extracted], '.-', label='# of violating instances (count)')
    ax.grid()
    ax.set_xlabel(r'Time [$sec$]')
    # ax.set_ylabel('Distance (m)')
    ax.set(xlim=(0, t_max))
    ax.legend(loc=1)

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(ts[extracted], density[extracted], '.-', label=r'social density $\rho$ (ped./$m^2$)')
    ax.grid()
    ax.set_xlabel(r'Time [$sec$]')
    # ax.set_ylabel('Density (ped./m^2)')
    ax.set(xlim=(0, t_max))
    ax.legend(loc=1)

    fig.savefig(os.path.join(path_analysis, dataset + '_statistics_vs_time.png'))
    plt.close(fig)

    # ============= below are figures of 2d histograms ================
    bin_size = 15

    # ---- figure 4 ----
    fig = plt.figure(figsize=(4.15, 3.27))
    fig.subplots_adjust(left=0.18, right=1.0, top=0.9, bottom=0.14)

    fig.suptitle(dict_dataset_names[dataset])
    ax = fig.add_subplot(1, 1, 1)
    plt.hist2d(avg_min_dists, density,
               bins=(bin_size, bin_size),
               # range=[[0.0, 6.0], [0.0, 0.15]],
               # norm=LogNorm()
               )
    plt.colorbar()
    ax.set_ylabel(r'Social Density $\rho$ (ped./$m^2$)')
    ax.set_xlabel('Avg. Closest Physical Distance $d_{avg}$ ($m$)')
    fig.savefig(os.path.join(path_analysis, dataset + '_2d_hist_density_vs_avg_dists.png'))
    plt.close(fig)

    # ---- figure 5 ----
    fig = plt.figure(figsize=(4.15, 3.27))
    fig.subplots_adjust(left=0.18, right=1.0, top=0.9, bottom=0.14)
    fig.suptitle(dict_dataset_names[dataset])
    ax = fig.add_subplot(1, 1, 1)
    plt.hist2d(min_min_dists, density,
               bins=(bin_size, bin_size),
               # range=[[0.0, 6.0], [0.0, 0.15]],
               # norm=LogNorm()
               )
    plt.colorbar()
    ax.set_ylabel(r'Social Density $\rho$ (ped./$m^2$)')
    ax.set_xlabel(r'Min. Closest Physical Distance $d_{min}$ ($m$)')
    fig.savefig(os.path.join(path_analysis, dataset + '_2d_hist_density_vs_min_dists.png'))
    plt.close(fig)

    # ---- figure 6 ----
    fig = plt.figure(figsize=(4.15, 3.27))
    fig.subplots_adjust(left=0.18, right=1.0, top=0.9, bottom=0.14)
    fig.suptitle(dict_dataset_names[dataset])
    ax = fig.add_subplot(1, 1, 1)
    plt.hist2d(violations, density,
               bins=(bin_size, bin_size),
               # range=[[0.0, 6.0], [0.0, 0.15]],
               # norm=LogNorm()
               )
    plt.colorbar()
    ax.set_ylabel(r'Social Density $\rho$ (ped./$m^2$)')
    # ax.set_yticks(np.arange(0, 0.15, 0.01))
    ax.set_xlabel(r'Num. of Social Distancing Violations $v$')
    fig.savefig(os.path.join(path_analysis, dataset + '_2d_hist_density_vs_violation.png'))
    plt.close(fig)

    # ---- figure 7 ----
    fig = plt.figure(figsize=(3.28, 3.14))
    fig.subplots_adjust(left=0.21, bottom=0.15, right=0.97, top=0.90)
    fig.suptitle(dict_dataset_names[dataset])

    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.plot(violations, density, '.b', alpha=0.2)
    ax.set_ylabel(r'Social Density $\rho$ (ped./$m^2$)')
    ax.set_xlabel(r'Num. of Social Distancing Violations $v$')

    print('Avg. Avg. Closest Physical Distance = %.6f' % np.mean(avg_min_dists))
    print('Skewness = %.6f' % stats.skew(density))

    intercept, slope, preds, lbs, ubs, x_select, lb_select, ub_select = \
        custom_simple_linear_regression(xs=violations, ys=density, x_select='y_intercept')

    print('x_select = %.6f' % x_select)
    print('x_select_lb = %.6f' % lb_select)
    print('x_select_ub = %.6f' % ub_select)

    line = [0, max(violations)]
    # line = [-0.01, 0.175]

    ax.plot(line, [intercept + slope * line[0], intercept + slope * line[1]], '-r')
    ax.plot(preds, lbs, '-g')
    ax.plot(preds, ubs, '-g')
    # ax.plot(0.0, lb_select, '.r')
    plt.text(0.0 + 0.5, lb_select - 0.005, r'$\rho_c$', fontsize=15, color='r')
    ax.set(xlim=(0.0, np.max(violations)))
    ax.set(ylim=(0.0, np.max(density)))
    # ax.vlines(0, -0.01, 0.15)
    # ax.hlines(0, line[0], line[1])
    # w = 1/60*density_max
    # ax.plot([-w, w], [ci[1], ci[1]], color='r')
    # ax.plot([-w, w], [ci[0], ci[0]], color='r')
    # ax.plot([0, 0], ci, color='r')

    # ax.hlines(intercept - std_dev, -w, w, color='r')
    # ax.vlines(0, intercept - std_dev, intercept + std_dev, color='r')
    # plt.show()

    fig.savefig(os.path.join(path_analysis, dataset + '_regression_density_vs_violation.png'))
    plt.close(fig)


def main():
    for dataset in ['oxford_town', 'mall', 'grand_central']:
    # for dataset in ['oxford_town', 'mall']:

        analyze_statistics(dataset=dataset)


if __name__ == '__main__':
    main()