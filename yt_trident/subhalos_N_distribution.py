#!/usr/bin/env python
import matplotlib.pyplot as plt
plt.ion()
from subhalos_analysis import subhalo_analysis_one_to_map
import glob
import pandas as pd
from multiprocessing import Pool
from tools import usable_cores, subhalo_virial_radius, HUBBLE_2Mpc_LG, \
                  subhalo_center, absorber_region_2Mpc_LG

abs_directory = './absorbers_2Mpc_LG_from_subhalos_fuzzy/'
pool = Pool(usable_cores())

lines_abs = ['Si III 1206', 'Si II 1193', 'C II 1335', 'C IV 1548']

def as_si(x, ndp):
    """
    to render exponential labels in latex
    taken from https://stackoverflow.com/a/31453961
    """
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))


def subhalos_analysis_parallel(abs_directory, pool):

    tasks = [handle for handle in glob.glob(abs_directory + 'abs*')]

    results = pool.map(subhalo_analysis_one_to_map, tasks)

    return pd.concat(results).sort_values(['mass', 'sub', 'b'], ascending=False)


def plot_N_vs_b(index, axarr, data):

    masses = data['mass'].unique()
    data_now = data [data['mass'] == masses[index]]

    subhalo_number = data_now['sub'].iloc[0]
    radius = subhalo_virial_radius(subhalo_number)
    sub_pos = subhalo_center(subhalo_number)
    region = absorber_region_2Mpc_LG(sub_pos)


    for line in lines_abs:
        axarr[index].semilogy(data_now['b']/radius, data_now[line], label = '')


    axarr[index].plot([], [], ' ', label="$M= {0:s}$".format(
                                as_si(data_now['mass'].iloc[0],2))+" $M_{sun}$")

    axarr[index].plot([], [], ' ', label="$R_{vir}$"+"$= {:.2f}$ kpc".format(
                                                        radius/HUBBLE_2Mpc_LG))

    axarr[index].plot([], [], ' ', label="Region: {}".format(region))


    axarr[index].fill_between(x=[0, 2], y1 = 10**12, y2 = 10**14, color = 'lavender')
    axarr[index].grid()
    axarr[index].set_xlim(0,2)
    axarr[index].set_ylim(10**5, 10**17)
    axarr[index].legend(markerscale=0)

    if index in [9, 10, 11]:
        axarr[index].set_xlabel('$b \\slash R_{vir}$', fontsize = 15)

    if index in [0, 3, 6, 9]:
        axarr[index].set_ylabel('$N$ [cm$^{-2}$]', fontsize = 15)

    # if index == 1:
    #     axarr[index].set_title('Column density by impact parameter in subhalos',
    #                             fontsize=17)


if __name__ == '__main__':

    # data = subhalos_analysis_parallel(abs_directory, pool)

    try:
        data = pd.read_pickle('./subhalos_N_b.pkl')

    except FileNotFoundError:
        print('Pickle not found')
        data = subhalos_analysis_parallel(abs_directory, pool)


    fig, axarr = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True)
    axarr = axarr.flatten()

    for index in range(len(axarr)):

        plot_N_vs_b(index, axarr, data)

    fig.subplots_adjust(wspace=0, hspace=0)
    # fig.suptitle('Column density by impact parameter in subhalos', fontsize=20)
    # fig.legend()
