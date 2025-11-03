import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time

from hallett.nltsim.core import *
from hallett.nltsim.analysis import *
from system_example1 import *

parser = argparse.ArgumentParser()
parser.add_argument("--f0", type=float, default=8e9, help="Stimulus frequency [Hz]")
parser.add_argument("--V0", type=float, default=0.5, help="Stimulus amplitude [V]")
parser.add_argument("--Vmin", type=float, default=-1, help="Min total length [m]")
parser.add_argument("--Vmax", type=float, default=1, help="Max total length [m]")
parser.add_argument("--num_sweep", type=int, default=9, help="Number of sweep points")
parser.add_argument("--dx_ref", type=float, default=None,
				help="Reference spatial step [m]. If None, use 0.3/600 for FDTD, 0.24/240 for Ladder")
parser.add_argument("--T", type=float, default=4e-9, help="Total simulation time [s]")
parser.add_argument("-p", "--parallel", action='store_true', help="Run simulations in parallel")
parser.add_argument("--tail", type=float, default=1.5e-9, help="Analyze only last 'tail' seconds")
parser.add_argument("--bw_bins", type=int, default=3, help="Half-width in FFT bins for harmonic integration")
parser.add_argument("--out", type=str, default="sweep_length.csv", help="Output CSV path")
parser.add_argument("--plot", action="store_true", help="Plot P(dBm) vs L for harmonics")
parser.add_argument("--csv", action="store_true", help="Save result to CSV")
parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel workers for the sweep (use -1 for all cores)")

args = parser.parse_args()

# ---------------------------- Main ----------------------------

def main():
	
	
	if args.dx_ref is None:
		args.dx_ref = 5e-4
	
	bias_vals = np.linspace(args.Vmin, args.Vmax, args.num_sweep)
	
	fdtd_exp_powers = HarmonicPowers(x_parameter="Bias voltage", x_values=bias_vals)
	fdtd_imp_powers = HarmonicPowers(x_parameter="Bias voltage", x_values=bias_vals)
	le_exp_powers = HarmonicPowers(x_parameter="Bias voltage", x_values=bias_vals)
	le_imp_powers = HarmonicPowers(x_parameter="Bias voltage", x_values=bias_vals)
	
	phys_length = 0.2
	
	t0 = time.time()
	print(f"Running sequentially.")
	
	# Scan over all lengths
	for vb in bias_vals:
		
		# Define system parameters
		fdtd_params_exp, le_params_exp = define_system(phys_length, args.f0, args.V0, args.T, args.dx_ref, False, V_bias=vb) # Explicit params
		fdtd_params_imp, le_params_imp = define_system(phys_length, args.f0, args.V0, args.T, args.dx_ref, True, V_bias=vb) # Implicit params
		
		fdtd_exp_sim = FiniteDiffSim(fdtd_params_exp) # Create sim
		fdtd_exp_out = fdtd_exp_sim.run() # run
		load_harmonics_probe(fdtd_exp_out, fdtd_exp_powers, args.f0, args.tail, args.bw_bins, 3)
		
		le_exp_sim = LumpedElementSim(le_params_exp) # Create sim
		le_exp_out = le_exp_sim.run() # Run
		load_harmonics_probe(le_exp_out, le_exp_powers, args.f0, args.tail, args.bw_bins, 3)
		
		fdtd_imp_sim = FiniteDiffSim(fdtd_params_imp) # Create sim
		fdtd_imp_out = fdtd_imp_sim.run() # run
		load_harmonics_probe(fdtd_imp_out, fdtd_imp_powers, args.f0, args.tail, args.bw_bins, 3)
		
		le_imp_sim = LumpedElementSim(le_params_imp) # Create sim
		le_imp_out = le_imp_sim.run() # Run
		load_harmonics_probe(le_imp_out, le_imp_powers, args.f0, args.tail, args.bw_bins, 3)
	
	print(f"Sequential simulation finished ({time.time()-t0} sec).")
	
	c_fund = 'tab:blue'
	c_2h = 'tab:orange'
	c_3h = 'tab:green'
	
	c_fundL = 'tab:cyan'
	c_2hL = 'tab:purple'
	c_3hL = 'tab:gray'
	
	plt.figure(1, figsize=(8,5))
	
	plt.plot(bias_vals, fdtd_exp_powers.f0, marker='x', label='Fundamental, Explicit', color=c_fund, linestyle='--')
	plt.plot(bias_vals, fdtd_exp_powers.h2, marker='x', label='2nd harmonic, Explicit', color=c_2h, linestyle='--')
	plt.plot(bias_vals, fdtd_exp_powers.h3, marker='x', label='3rd harmonic, Explicit', color=c_3h, linestyle='--')
	
	plt.plot(bias_vals, fdtd_imp_powers.f0, marker='+', label='Fundamental, Implicit', color=c_fund, linestyle=':')
	plt.plot(bias_vals, fdtd_imp_powers.h2, marker='+', label='2nd harmonic, Implicit', color=c_2h, linestyle=':')
	plt.plot(bias_vals, fdtd_imp_powers.h3, marker='+', label='3rd harmonic, Implicit', color=c_3h, linestyle=':')
	
	plt.xlabel("Bias Voltage (V)")
	plt.ylabel("Power at load (dBm)")
	plt.title(f"FDTD Simulation, Explicit vs Implicit Updates")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	
	plt.figure(2, figsize=(8,5))
	
	plt.plot(bias_vals, le_exp_powers.f0, marker='x', label='Fundamental, Explicit', color=c_fundL, linestyle='--')
	plt.plot(bias_vals, le_exp_powers.h2, marker='x', label='2nd harmonic, Explicit', color=c_2hL, linestyle='--')
	plt.plot(bias_vals, le_exp_powers.h3, marker='x', label='3rd harmonic, Explicit', color=c_3hL, linestyle='--')
	
	plt.plot(bias_vals, le_imp_powers.f0, marker='+', label='Fundamental, Implicit', color=c_fundL, linestyle=':')
	plt.plot(bias_vals, le_imp_powers.h2, marker='+', label='2nd harmonic, Implicit', color=c_2hL, linestyle=':')
	plt.plot(bias_vals, le_imp_powers.h3, marker='+', label='3rd harmonic, Implicit', color=c_3hL, linestyle=':')
	
	plt.xlabel("Bias Voltage (V)")
	plt.ylabel("Power at load (dBm)")
	plt.title(f"Lumped-Element Simulation, Explicit vs Implicit Updates")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	
	plt.figure(3, figsize=(8,5))
	
	plt.plot(bias_vals, fdtd_imp_powers.f0, marker='x', label='Fundamental, FDTD', color=c_fund, linestyle='--')
	plt.plot(bias_vals, fdtd_imp_powers.h2, marker='x', label='2nd harmonic, FDTD', color=c_2h, linestyle='--')
	plt.plot(bias_vals, fdtd_imp_powers.h3, marker='x', label='3rd harmonic, FDTD', color=c_3h, linestyle='--')
	
	plt.plot(bias_vals, le_imp_powers.f0, marker='+', label='Fundamental, Lumped-Element', color=c_fundL, linestyle=':')
	plt.plot(bias_vals, le_imp_powers.h2, marker='+', label='2nd harmonic, Lumped-Element', color=c_2hL, linestyle=':')
	plt.plot(bias_vals, le_imp_powers.h3, marker='+', label='3rd harmonic, Lumped-Element', color=c_3hL, linestyle=':')
	
	plt.xlabel("Bias Voltage (V)")
	plt.ylabel("Power at load (dBm)")
	plt.title(f"Implicit Updates, FDTD vs Lumped-Element Simulations")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	
	
	plt.show()

if __name__ == "__main__":
	main()
