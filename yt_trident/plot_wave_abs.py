# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
from tools import get_line

absorber_filename = 'abs_990_0.70_4.19.txt'
spectrum_filename = './spectra_C_Si_2Mpc_LG_from_mw/spec_990_0.70_4.19.txt'

line = 'C II'
key = 'C II* 1335'

wavelength, tau, flux, flux_error = np.loadtxt(fname=spectrum_filename, delimiter=' ',
                                    skiprows=1, unpack=True)

df = pd.read_csv(absorber_filename, skiprows=1)

wave_section, flux_section = get_line(line, wavelength, flux, wavelength_interval=16)

line_df = pd.concat((df[df['Line']=='C II 1335'], df[df['Line']=='C II* 1336']))
line_df = line_df.sort_values('lambda')

plt.figure()
plt.plot(wave_section, flux_section, 'b')
plt.plot(line_df['lambda'], np.exp(-line_df['tau']),'r')
