# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()
import yt
import trident
from tools import get_line, line_table
from R_N_fig1 import load_or_make_spectrum, plot_line


distances = np.linspace(0, 1000, 100)
