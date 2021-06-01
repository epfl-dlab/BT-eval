from collections import defaultdict
import numpy as np
import scipy.stats as stats
import pandas as pd
from bt import BT
from Elo import ELO
from TrueSkill import TrueSkill
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def sample_dataset(n_regular_types, n_reverse_types, sys_strength, noise=1, n=100):
    systems_scores = defaultdict(list)
    for _ in range(n_regular_types):
        scale = 100 * np.random.rand()
        for sys, alpha in sys_strength.items():
            n_samples = int(n / n_regular_types)
            systems_scores[sys].extend(stats.norm.rvs(loc=alpha + scale, scale=noise, size=n_samples))

    # for _ in range(n_reverse_types):
    #     scale = 100 * np.random.rand()
    #     for sys, alpha in sys_strength.items():
    #         inverse_perf = 100 - alpha
    #         systems_scores[sys].extend(stats.norm.rvs(loc=inverse_perf + scale, scale=noise, size=n))

    return pd.DataFrame.from_dict(systems_scores)


def add_outliers(df, sys_strengths, percent_outliers):
    row_list = []
    if percent_outliers > 0:
        n_outliers = max(int(percent_outliers * df.shape[0]), 1)
    else:
        n_outliers = int(percent_outliers * df.shape[0])
    strengths = list(sys_strengths.values())
    while len(row_list) < n_outliers:
        observations = np.random.rand(len(sys_strengths.keys()))
        if stats.kendalltau(observations, strengths)[0] < 0.:
            row_list.append(dict(zip(sys_strengths.keys(), 100 * observations)))

    return pd.concat([df, pd.DataFrame(row_list)], axis=0)


def evaluate(df, sys_strength, method):
    mean_scores = dict(zip(df.columns, df.mean(axis=0).to_list()))
    median_scores = dict(zip(df.columns, df.median(axis=0).to_list()))
    if method['name'] == 'BT':
        bt_scores = dict(zip(df.columns, BT(df)))
    elif method['name'] == 'ELO':
        bt_scores = dict(zip(df.columns, ELO(df, method['k'])))
    else:
        bt_scores = dict(zip(df.columns, TrueSkill(df, method['mu'], method['sigma'], method['beta'])))

    bt, mean, median = [], [], []
    for s in sys_strength.keys():
        bt.append(bt_scores[s])
        median.append(median_scores[s])
        mean.append(mean_scores[s])
    strengths = list(sys_strength.values())

    return stats.kendalltau(strengths, mean)[0], stats.kendalltau(strengths, median)[0], stats.kendalltau(strengths, bt)[0]


def run_simulations(n_regular_list, percentage_reverse, percent_outliers_list, n_systems_list, n_samples_list, method='BT'):
    n_repeat = 10
    mean_perf, median_perf, bt_perf = [], [], []
    number_samples, number_regular_types, percent_outliers, number_reverse_types, noise, n_systems = [], [], [], [], [], []
    for n_reg in n_regular_list:
        for rev_percent in percentage_reverse:
            n_rev = int(rev_percent * n_reg)  # + 1
            for outlier_percent in percent_outliers_list:
                for n_sys in n_systems_list:
                    for n_samples in n_samples_list:
                        for _ in range(n_repeat):
                            strengths = np.random.rand(n_sys)
                            strengths /= np.sum(strengths)
                            sys_strengths = dict(zip(['sys_{}'.format(i) for i in range(n_sys)], 10 * strengths))
                            dataset = sample_dataset(n_reg, n_rev, sys_strengths, n=n_samples)
                            dataset = add_outliers(dataset, sys_strengths, outlier_percent)
                            # print(dataset)
                            # exit()
                            res = evaluate(dataset, sys_strengths, method=method)
                            mean, median, bt = res

                            mean_perf.append(mean)
                            median_perf.append(median)
                            bt_perf.append(bt)
                            percent_outliers.append(outlier_percent)
                            number_samples.append(dataset.shape[0])
                            number_regular_types.append(n_reg)
                            number_reverse_types.append(rev_percent)
                            noise.append(0.1)
                            n_systems.append(n_sys)

    return pd.DataFrame.from_dict({'Mean': mean_perf, 'Median': median_perf, 'BT': bt_perf,
                                   'n_samples': number_samples, 'n_regular': number_regular_types, 'n_outliers': percent_outliers, 'n_reverse': number_reverse_types,
                                   'noise': noise, 'n_systems': n_systems})


def obtain_x_y_yerr(df, name_x, name_y):
    x = df.groupby(name_x).mean().index.to_list()
    y = df.groupby(name_x).mean()[name_y].to_list()
    # m = df.groupby(name_x).quantile(0.10)[name_y]
    # M = df.groupby(name_x).quantile(0.90)[name_y]
    # yerr = (M - m) / 2.
    yerr = 2 * 1.96 * df.groupby(name_x).sem()[name_y].to_numpy()
    return x, y, yerr


if __name__ == '__main__':
    n_regular_list = [1, 3, 5, 10]
    percentage_reverse = [0.]
    n_systems_list = [2, 3, 5, 10, 25, 50]
    percent_outliers_list = [0., 0.01, 0.025, 0.05, 0.075]
    n_samples = [10, 30, 100, 200]
    res_df = run_simulations(n_regular_list, percentage_reverse, percent_outliers_list, n_systems_list, n_samples)

    print(res_df.mean(axis=0))
    easy_cases = res_df[(res_df['n_outliers'] == 0.) & (res_df['n_reverse'] == 0.)]
    easy_cases = easy_cases[easy_cases['n_regular'] == 1]

    fig, axes = plt.subplots(1, 6, figsize=(30, 5), sharey=True)
    ft = 15
    ax = axes[0]
    x_axis = 'n_systems'
    x, y, yerr = obtain_x_y_yerr(easy_cases, x_axis, 'Mean')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:blue')
    x, y, yerr = obtain_x_y_yerr(easy_cases, x_axis, 'Median')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:green')
    x, y, yerr = obtain_x_y_yerr(easy_cases, x_axis, 'BT')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:red')
    ax.set_xlabel('Number of systems (easy cases)', fontsize=ft)
    ax.set_ylabel('Kendall\'s tau with true strengths', fontsize=ft)

    # ax.set_ylabel('Kendall\'s tau with true strengths', fontsize=17)

    # res_df['difficulty'] = res_df['n_outliers']  # + res_df['n_reverse']
    ax = axes[1]
    outliers_df = res_df[res_df['n_regular'] == 1]
    x_axis = 'n_outliers'
    x, y, yerr = obtain_x_y_yerr(outliers_df, x_axis, 'Mean')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:blue')
    x, y, yerr = obtain_x_y_yerr(outliers_df, x_axis, 'Median')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:green')
    x, y, yerr = obtain_x_y_yerr(outliers_df, x_axis, 'BT')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:red')
    ax.set_xlabel('Percentage of outliers (1 test type)', fontsize=ft)
    # ax.set_ylabel('Kendall\'s tau with true strengths', fontsize=18)

    ax = axes[2]
    regular_df = res_df[res_df['n_outliers'] == 0.]
    x_axis = 'n_regular'
    x, y, yerr = obtain_x_y_yerr(regular_df, x_axis, 'Mean')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:blue')
    x, y, yerr = obtain_x_y_yerr(regular_df, x_axis, 'Median')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:green')
    x, y, yerr = obtain_x_y_yerr(regular_df, x_axis, 'BT')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:red')
    ax.set_xlabel('Test instances types (no outliers)', fontsize=ft)
    # ax.set_ylabel('Kendall\'s tau with true strengths', fontsize=18)

    ax = axes[3]
    # regular_df = res_df[res_df['n_outliers'] == 0.]
    x_axis = 'n_regular'
    x, y, yerr = obtain_x_y_yerr(res_df, x_axis, 'Mean')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:blue')
    x, y, yerr = obtain_x_y_yerr(res_df, x_axis, 'Median')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:green')
    x, y, yerr = obtain_x_y_yerr(res_df, x_axis, 'BT')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:red')
    ax.set_xlabel('Test instances types (with outliers)', fontsize=ft)

    ax = axes[4]
    # outliers_df = res_df[res_df['n_regular'] == 1]
    x_axis = 'n_outliers'
    x, y, yerr = obtain_x_y_yerr(res_df, x_axis, 'Mean')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:blue')
    x, y, yerr = obtain_x_y_yerr(res_df, x_axis, 'Median')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:green')
    x, y, yerr = obtain_x_y_yerr(res_df, x_axis, 'BT')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:red')
    ax.set_xlabel('Percentage of outliers (varying test types)', fontsize=ft)

    ax = axes[5]
    x_axis = 'n_systems'
    x, y, yerr = obtain_x_y_yerr(res_df, x_axis, 'Mean')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:blue')
    x, y, yerr = obtain_x_y_yerr(res_df, x_axis, 'Median')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:green')
    x, y, yerr = obtain_x_y_yerr(res_df, x_axis, 'BT')
    ax.errorbar(x, y, yerr, elinewidth=2, linewidth=2, fmt='-o', ms=7, color='tab:red')
    ax.set_xlabel('Number of systems (all cases)', fontsize=ft)

    legend_elem = [Line2D([0], [0], linestyle='-', linewidth=2, c='tab:blue', label='Mean'),
                   Line2D([0], [0], linestyle='-', linewidth=2, c="tab:green", label='Median'),
                   Line2D([0], [0], linestyle='-', linewidth=2, c="tab:red", label='BT')]

    fig.legend(handles=legend_elem, ncol=3, loc='upper center', frameon=False, fontsize=21, bbox_to_anchor=(0.55, 1.15))
    #fig.legend(handles=legend_elem, fontsize=13)
    fig.tight_layout(pad=1.1)
    # fig.savefig("simulations.pdf", bbox_inches="tight")

    plt.show()
    # Plot for increasing n_systems for easy cases

    # Plot with increasing difficulty: outliers + rev

    # Plot for increasing n_system all cases

    # Plot for increasing noise easy cases

    # Plot for increasing noise all cases
