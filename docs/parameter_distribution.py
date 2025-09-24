import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def default_settings(lw=1, column=2, width=2.6 * 3, length=3, ts=9, ls=9, fs=9):
    if column == 1.5:
        width = 4.5
    elif column == 1:
        widdefault_settingsth = 3.42
    elif column == 2:
        width = 7
    plt.rcParams['figure.figsize'] = (width, length)
    plt.rcParams['figure.facecolor'] = 'none'
    plt.rcParams['font.size'] = fs
    plt.rcParams['axes.facecolor'] = 'none'
    plt.rcParams['axes.titlesize'] = ts
    plt.rcParams['axes.labelsize'] = ls
    plt.rcParams['lines.linewidth'] = lw
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['legend.loc'] = 'upper right'
    plt.rcParams["legend.frameon"] = False


def plt_model_params():
    #plot_style()
    model_cells = pd.read_csv("models_big_fit.csv")
    #plt.hist(model_cells)
    model_cells.pop('v_zero')
    model_cells.pop('v_base')
    model_cells.pop('threshold')
    model_cells.pop('cell')
    model_cells.pop('a_zero')
    model_cells.pop('deltat')
    model_cells.pop('EODf')
    model_cells.median()

    #plot_style()
    default_settings(ls=10, ts=10, fs=8, column=2, length=3)
    fig, ax = plt.subplots(2,4, constrained_layout = True, sharey = True)
    ax = np.concatenate(ax)
    keys = np.array(model_cells.keys())
    keys = ['dend_tau',
            'input_scaling',
            'delta_a',
            'tau_a',
            'noise_strength',
            'v_offset',
            'mem_tau',
           'ref_period', ]
    titles = [r'$\tau_{d}$ [s]',
              r'$\alpha$',
              r'$\Delta_{A}$',
              r'$\tau_{a}$ [s]',
              r'$\sqrt{2D}$',
              r'$\mu$',
              r'$\tau_{m}$ [s]',r'$t_{ref}$ [s]']#r'$\Delta t$ [s]',
    for a in range(len(model_cells.keys())):
        ax[a].hist(np.array(model_cells[keys[a]]), bins = 20)
        #if '[s]' in titles[a]:
        ax[a].axvline(model_cells[keys[a]].median(), color = 'black', zorder = 2)
        ax[a].set_xlabel(titles[a])
    ax[0].set_ylabel('count')
    ax[4].set_ylabel('count')
    plt.savefig('parameter_distribution.pdf')
    plt.close()

if __name__ == '__main__':
    plt_model_params()