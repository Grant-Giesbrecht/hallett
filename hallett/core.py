import numpy as np
import os
from stardust.io import hdf_to_dict
import csv
import re


def lin_to_dB(x_lin:float, use10:bool=False):
	if use10:
		return 10*np.log10(x_lin)
	else:
		return 20*np.log10(x_lin)

def has_ext(path, exts):
	return os.path.splitext(path)[1].lower() in [e.lower() for e in exts]

def bounded_interp(x, y, x_target):
	if x_target < x[0] or x_target > x[-1]:
		return None
	return np.interp(x_target, x, y) 
