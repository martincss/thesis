#!/usr/bin/env python
import matplotlib.pyplot as plt
plt.ion()
from subhalos_analysis import subhalo_analysis_one_to_map
import glob
import pandas as pd
from multiprocessing import Pool
from tools import usable_cores

abs_directory = './absorbers_2Mpc_LG_from_subhalos_fuzzy/'
pool = Pool(usable_cores())

def subhalos_analysis_parallel(abs_directory, pool):

    tasks = [handle for handle in glob.glob(abs_directory + 'abs*')]

    results = pool.map(subhalo_analysis_one_to_map, tasks)

    return pd.concat(results).sort_values(['mass', 'sub', 'b'], ascending=False)


if __name__ == '__main__':
    data = subhalos_analysis_parallel(abs_directory, pool)
