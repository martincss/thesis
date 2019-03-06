#!/usr/bin/env python
import matplotlib.pyplot as plt
plt.ion()
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
from tools import HUBBLE_2Mpc_LG
from profiles_mw import covering_fractions_2k
from profiles_m31 import covering_fractions_m31

abs_lines = ['Si III 1206', 'Si II 1193', 'C II 1335', 'C IV 1548']

R_BIN_LENGTH = 3 # kpc/h

exp1 = 10
exp2 = 13

thresh_1 = 10**exp1
thresh_2 = 10**exp2

def plot_covfs(r, covfs):

    mw1, mw2, m311, m312 = covfs

    fig, axarr = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

    # this hurt me more than it did to you
    for line, covf in mw1.items():
        axarr[0,0].plot(r[:-1], covf, label = line)

    for line, covf in mw2.items():
        axarr[1,0].plot(r[:-1], covf, label = line)

    for line, covf in m311.items():
        axarr[0,1].plot(r[:-1], covf, label = line)

    for line, covf in m312.items():
        axarr[1,1].plot(r[:-1], covf, label = line)

    axarr[1,0].set_xlabel('Distance [kpc]', fontsize = 15)
    axarr[1,1].set_xlabel('Distance [kpc]', fontsize = 15)
    axarr[0,0].set_ylabel('$f$  with $\\log N > $ {:d}'.format(exp1), fontsize = 15)
    axarr[0,0].set_title('MW', fontsize = 15)
    axarr[0,1].set_title('M31', fontsize = 15)
    axarr[0,1].legend()
    axarr[1,0].set_ylabel('$f$  with $\\log N > $ {:d}'.format(exp2), fontsize = 15)

    for ax in axarr.flatten():
        ax.grid()

    fig.suptitle('Covering fraction by distance', fontsize = 20)
    fig.subplots_adjust(wspace=0, hspace=0)

    return fig, axarr


if __name__ == '__main__':

    print('first covering fractions')
    r, covfs_mw_first = covering_fractions_2k(abs_lines, thresh_1, R_BIN_LENGTH)
    r, covfs_m31_first = covering_fractions_m31(abs_lines, thresh_1,R_BIN_LENGTH)

    print('next covering fractions')
    r, covfs_mw_second = covering_fractions_2k(abs_lines, thresh_2, R_BIN_LENGTH)
    r, covfs_m31_second = covering_fractions_m31(abs_lines, thresh_2,R_BIN_LENGTH)

    r /= HUBBLE_2Mpc_LG

    fig, axarr = plot_covfs(r,[covfs_mw_first, covfs_mw_second,
                               covfs_m31_first, covfs_m31_second])
