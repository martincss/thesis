# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
plt.ion()
import yt
yt.enable_parallelism()
import trident
from tools import make_SpectrumGenerator, get_line_observables_dict
import pandas as pd




def get_lines(line_observables_dict):

    lines = sorted(list(line_observables_dict.keys()))

    return lines
