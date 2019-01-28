# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd


abs_m31_filename = './absorbers_2Mpc_LG_to_mw_2000_wrt_mwcenter/abs_2000.000_0.698_4.115.txt'

df_m31 = pd.read_csv(abs_m31_filename, skiprows=1)

line = 'Si III 1206'
dfs = df_m31[ df_m31['Line'] == line ]


if __name__=='__main__':

    fig, axs = plt.subplots(3,1, sharex=True)

    axs[0].scatter(dfs['z_cosmo'], 1+dfs['N'])
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Column density')
    axs[0].set_xlim(-0.0007, -0.0004)

    axs[1].plot(dfs['z_cosmo'], dfs['v_los'])
    axs[1].set_ylabel('v los')

    # axs[2].scatter(dfs['z_eff'], dfs['N'])
    # axs[2].set_yscale('log')
    # axs[2].set_ylabel('Column density')

    axs[2].plot(dfs['z_cosmo'], dfs['lambda'] - dfs['delta_lamdba']/(1+dfs['z_dopp']))
    axs[2].set_label('lamba with only z_cosmo contribution')
