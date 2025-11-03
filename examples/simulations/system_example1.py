import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

from hallett.nltsim.core import *
from hallett.nltsim.analysis import *

def define_system(L:float, f0:float, V0:float, T:float, dx_ref:float, implicit:bool, V_bias:float=0):
	''' Returns parameter objects for the two sim types. Defined in a function so it's easier
	to recycle the same system into multiple simulations.
	
	Parameters:
		L (float): Length overall of system. All regions scale with L.
		f0 (float): Fundamental tone in Hz
		T (float):
	
	'''
	
	dist_L0 = 1e-6
	dist_C0 = 130e-12
	
	# Define source and load impedances
	Rs = 50.0
	RL = 50.0
	
	# Define regions
	regions = [
		TLINRegion(x0=0.0,     x1=L/4,   L0_per_m=dist_L0, C_per_m=dist_C0, alpha=2.0e-3),
		TLINRegion(x0=L/4,     x1=3*L/4, L0_per_m=dist_L0, C_per_m=dist_C0, alpha=100),
		TLINRegion(x0=3*L/4,   x1=L,     L0_per_m=dist_L0, C_per_m=dist_C0, alpha=1.0e-3),
	]
	
	# Select a ∆t using the CFL condition
	Nx = max(50, int(np.round(L / dx_ref))) 
	dx = L / Nx # Get ∆x
	Lmin = min(r.L0_per_m for r in regions) # Get max L
	Cmax = max(r.C_per_m for r in regions) # Get max C
	dt = FiniteDiffSim.cfl_dt(dx, Lmin, Cmax, safety=0.85) # Get CFL. Smaller `safety` makes smaller ∆t
	
	# Calculate omega
	w0 = 2*np.pi*f0
	
	# Define voltage stimulus
	Vs = lambda t: V0 * np.sin(w0 * t) + V_bias
	
	update_type = ("implicit" if implicit else "explicit")
	# print(f"update = {update_type}")
	
	N_lumped = max(30, int(np.round(L / dx_ref)))
	dt_ladder = 2.0e-12  # fixed ladder dt (edit if needed)
	
	# Prepare parameter object for FDTD simulation
	fdtd_params = FiniteDiffParams(Nx=Nx, L=L, dt=dt, T=T, Rs=Rs, RL=RL, Vs_func=Vs, regions=regions, nonlinear_update=update_type)
	
	# Prepare parameter object for Lumped element simulation
	le_params = LumpedElementParams(N=N_lumped, L=L, Rs=Rs, RL=RL, dt=dt_ladder, T=T, Vs_func=Vs, regions=regions, nonlinear_update=update_type)
	
	# Return param objects
	return fdtd_params, le_params