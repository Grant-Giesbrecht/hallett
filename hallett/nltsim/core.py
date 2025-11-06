from dataclasses import dataclass
import numpy as np
from typing import Callable, List, Literal, Tuple, Optional

def _newton_i_update(i0: np.ndarray, s: np.ndarray, L0: np.ndarray, alpha: np.ndarray, max_iter: int = 15, tol: float = 1e-12) -> np.ndarray:
	"""Solve per-element for i:  F(i) = i - i0 - s / (L0 * (1 + alpha*i^2)) = 0.

	Args:
		i0 (array): Previous current (same shape as s).
		s (array): s = dt * Δv  (ladder)   OR   s = - dt * (∂v/∂x) (FDTD)
		L0 (array): Base inductance and nonlinearity per element (section or half-cell).
		alpha (array): Base inductance and nonlinearity per element (section or half-cell).
	
	Returns:
		Returns updated estimates for current.
	"""
	# Initial guess: explicit update
	i = i0 + s / (L0 * (1.0 + alpha * i0**2))

	for _ in range(max_iter):
		denom = (1.0 + alpha * i**2)
		invLd = 1.0 / (L0 * denom)
		F = i - i0 - s * invLd
		# d(1/Ld)/di = -(1/L0) * (2*alpha*i) / (1 + alpha*i^2)^2
		dinvLd_di = -(1.0 / L0) * (2.0 * alpha * i) / (denom**2 + 1e-300)
		dF = 1.0 - s * dinvLd_di
		step = F / (dF + 1e-300)
		i_new = i - step
		if np.max(np.abs(step)) < tol:
			return i_new
		i = i_new
	return i

@dataclass
class LumpedElementResult:
	''' Class containing the result data from LumpedElementSim object.
	'''
	
	t: np.ndarray
	v_nodes: np.ndarray    # (Nt, N+1)
	i_L: np.ndarray        # (Nt, N)
	
	def probe_voltage(self, node: int) -> Tuple[np.ndarray, np.ndarray]:
		''' Gets the waveform at the specified node.
		
		Params:
			node (int): Node to probe
			
		Returns:
			tuple: (time, voltage) of the waveform at the node.
		'''
		
		t = np.asarray(self.t)
		node = int(node)
		v_t = np.asarray(self.v_nodes)[:, node]
		return t, v_t

@dataclass
class FiniteDiffResult:
	''' Class containing the result data from a FiniteDiffSim object.
	'''
	
	t: np.ndarray
	x: np.ndarray
	v_xt: np.ndarray   # (Nt, Nx+1)
	i_xt: np.ndarray   # (Nt, Nx)
	
	def probe_voltage(self, x: Optional[float] = None, index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
		''' Returns the waveform at the specified x position or index.
		
		Params:
			x (float): Optional, x position to probe.
			index (int): Optional, index to probe. Must specify `x` or `index`.
		
		Returns:
			(tuple): (time, voltage) of waveform at probe location
		'''
		
		t = np.asarray(self.t)
		if index is None:
			if x is None:
				raise ValueError("Provide either x or index for FDTD probe.")
			idx = int(np.argmin(np.abs(np.asarray(self.x) - x)))
		else:
			idx = int(index)
		v_t = np.asarray(self.v_xt)[:, idx]
		return t, v_t

@dataclass
class TLINRegion:
	''' Class used to describe a region of the simulation transmisison line.
	'''
	
	x0: float # start position (m)
	x1: float # End position (m)
	L0_per_m: float # Inductance per meter (H/m)
	C_per_m: float # Capacitance per meter (F/m)
	G_per_m: float # Conductance per meter (S/m)
	alpha: float # Nonlinearity (Model: L = L0*[1 + alpha*I**2] )

@dataclass
class LumpedElementParams:
	''' Class to describe basic parameters for a LumpedElementSim object.
	'''
	
	N: int # Number of ladder stages
	total_length: float # Total length (m)
	Rs: float # Source impedance (Ohms) (x=0)
	RL: float # Load impedance (Ohms)
	dt: float # Time step
	t_end: float # End time
	Vs_func: Callable[[float], float] # Source stimulus function ( V(t) )
	regions: List[TLINRegion] # List of TLINRegion objects describing transmission line
	nonlinear_update: Literal["explicit","implicit"] = "explicit" # Update method

@dataclass
class FiniteDiffParams:
	''' Class used to describe basic parameters for a FiniteDiffSim object.
	'''
	
	Nx: int # Number of discrete steps
	total_length: float # Total length (m)
	dt: float # Time step (s)
	t_end: float # End time (s)
	Rs: float # Source impedance (Ohms) (x=0)
	RL: float # Load impedance (Ohms) 
	Vs_func: Callable[[float], float] # Source stimulus function ( V(t) )
	regions: List[TLINRegion] # List of TLINRegion objects describing transmission line
	nonlinear_update: Literal["explicit","implicit"] = "explicit" # Update method

def _sample_regions_on_grid(regions: List, grid: np.ndarray, field: str) -> np.ndarray:
	''' Returns the selected parameter `field` sampled over the positions defined
	in `grid` from the applicable region in `regions`.
	
	Args:
		regions (list): List of TLINRegion objects defining the various regions
		grid (np.ndarray): Grid of positions on which to sample field.
		field (str): Name of parameter from TLINRegion which to sample.
	Returns:
		(np.ndarray): List of same shape as `grid`, containing the selected `field`
			from the correct region.
	'''
	
	# Initialize vals in same shape as the grid
	vals = np.zeros_like(grid, dtype=float)
	
	# Scan over each region
	for r in regions:
		
		# Get mask for which grid points region is present
		mask = (grid >= r.x0) & (grid < r.x1)
		
		# Save result to vals
		vals[mask] = getattr(r, field)
		
	# ensure last grid point gets last region's value
	end = max(r.x1 for r in regions)
	if np.isclose(grid[-1], end):
		for r in regions:
			if np.isclose(end, r.x1):
				vals[-1] = getattr(r, field)
	return vals

class LumpedElementSim:
	''' Simulator for L-C ladder based non-linear transmission line. '''
	
	def __init__(self, sim_params: LumpedElementParams):
		self.sim_params = sim_params
		self.Nt = int(np.round(sim_params.t_end / sim_params.dt)) + 1
		self.dx = sim_params.total_length / sim_params.N
		x_sec = (np.arange(sim_params.N) + 0.5) * self.dx
		x_nodes = np.arange(sim_params.N + 1) * self.dx
		self.L0_sec = _sample_regions_on_grid(sim_params.regions, x_sec, 'L0_per_m') * self.dx
		self.C_nodes = _sample_regions_on_grid(sim_params.regions, x_nodes, 'C_per_m') * self.dx
		self.G_nodes = _sample_regions_on_grid(sim_params.regions, x_nodes, 'G_per_m') * self.dx
		self.alpha_sec = _sample_regions_on_grid(sim_params.regions, x_sec, 'alpha')

	def run(self) -> LumpedElementResult:
		''' Runs the simulation.
		'''
		
		# Making local copies of variables for 
		sim_params = self.sim_params
		N, dt, Nt = sim_params.N, sim_params.dt, self.Nt
		
		# Initialize simulation parameters
		v = np.zeros(N+1) # Set voltage as zero everywhere
		iL_half = np.zeros(N) # Current flowing through inductors
		v_hist = np.zeros((Nt, N+1)) # Create a voltage history array
		i_hist = np.zeros((Nt, N)) # Create a current history array
		
		# Scan over all time points (as index)
		for n in range(Nt):
			
			t_n = n * dt # Get current iteration time
			Vs = sim_params.Vs_func(t_n) # Get current stimulus voltage
			
			# Get voltage delta across each ladder
			dv = v[:-1] - v[1:]
			
			# Update inductance and current estimates
			if sim_params.nonlinear_update == "explicit": # Explicit update
				
				# Estimate inductance from latest current estiamtes
				Ld = self.L0_sec * (1.0 + self.alpha_sec * iL_half**2)
				
				# Update latest current estimates by adding dt*dv/L
				iL_half = iL_half + dt * dv / Ld
			else: # Implicit update
				s = dt * dv
				iL_half = _newton_i_update(iL_half, s, self.L0_sec, self.alpha_sec) #TODO: Explain
			
			# Node updates
			dvdt = np.zeros_like(v)
			
			# Update first-node voltage from stimulus
			if sim_params.Rs == 0:
				v[0] = Vs
			else:
				# Update source current...
				if np.isinf(sim_params.Rs):
					i_src = 0.0
				else:
					i_src = (Vs - v[0]) / sim_params.Rs
				
				# Get voltage at first node from current
				dvdt[0] = (i_src - iL_half[0]) / self.C_nodes[0]
			
			# Update voltage change from current flowing into caps
			if N > 1:
				dvdt[1:-1] = (iL_half[:-1] - iL_half[1:]) / self.C_nodes[1:-1]
			
			# Calcualte load current
			if np.isinf(sim_params.RL):
				i_load = 0.0
			else:
				i_load = v[-1] / sim_params.RL
			
			# Update final dvdt point from load current
			dvdt[-1] = (iL_half[-1] - i_load) / self.C_nodes[-1]
			
			# Update voltage from dvdt and dt
			# v = v + dt * dvdt # Lossless model before G was added
			v = (v + dt * dvdt) / (1.0 + dt * self.G_nodes / self.C_nodes)
			
			# Add voltages and current to history parameters
			v_hist[n, :] = v
			i_hist[n, :] = iL_half
		
		# Create summary t array
		t = np.arange(Nt) * dt
		
		# Create data result object and return
		return LumpedElementResult(t=t, v_nodes=v_hist, i_L=i_hist)

class FiniteDiffSim:
	
	def __init__(self, sim_params: FiniteDiffParams):
		self.sim_params = sim_params
		self.dx = sim_params.total_length / sim_params.Nx
		self.Nt = int(np.round(sim_params.t_end / sim_params.dt)) + 1
		x_nodes = np.linspace(0.0, sim_params.total_length, sim_params.Nx + 1)
		x_half  = (np.arange(sim_params.Nx) + 0.5) * self.dx
		self.C_nodes = _sample_regions_on_grid(sim_params.regions, x_nodes, 'C_per_m') # C at nodes on fullsteps
		self.L0_half = _sample_regions_on_grid(sim_params.regions, x_half,  'L0_per_m') # L at nodes, offset by a halfstep
		self.G_nodes = _sample_regions_on_grid(sim_params.regions, x_nodes,  'G_per_m') # G at nodes on fullstep
		self.alpha_half = _sample_regions_on_grid(sim_params.regions, x_half, 'alpha') # Nonlinearity at nodes, offset by a halfset

	@staticmethod
	def cfl_dt(dx: float, Lmin: float, Cmin: float, safety: float = 0.9) -> float:
		''' Implements CFL condition to return a reasonable timestep to use, including
		a safety margin.
		'''
		v = 1.0 / np.sqrt(max(Lmin, 1e-300) * max(Cmin, 1e-300))
		return safety * dx / v

	def run(self) -> FiniteDiffResult:
		''' Run the simulation.
		'''
		
		# Create local copies of variables
		sim_params = self.sim_params
		Nx, dt, Nt = sim_params.Nx, sim_params.dt, self.Nt
		dx = self.dx
		
		# Initialize simulation data arrays
		v = np.zeros(Nx+1) # Voltages versus position
		i_half = np.zeros(Nx) # Current versus position, offset by a half step
		v_hist = np.zeros((Nt, Nx+1)) # Voltage history
		i_hist = np.zeros((Nt, Nx)) # Current history
		
		# Scan over all time points (as index, not time value)
		for n in range(Nt):
			
			t_n = n * dt # Get time value for this index
			Vs = sim_params.Vs_func(t_n) # Get stimulus voltage
			
			#  Get voltage different for each node
			dv_dx = (v[1:] - v[:-1]) / dx
			
			# Update latest current guess...
			if sim_params.nonlinear_update == "explicit": # Explicit guess
				
				# Update estimate for total inductance based on latest current estiamte
				Ld_half = self.L0_half * (1.0 + self.alpha_half * i_half**2)
				
				# Update estimate for current from inductance and dV/dt
				i_half = i_half - dt * dv_dx / Ld_half
			else: # Implicit guess
				s = - dt * dv_dx
				i_half = _newton_i_update(i_half, s, self.L0_half, self.alpha_half) #TODO: Explain
			
			# Estimate source current
			if sim_params.Rs == 0:
				i_left = None
			else:
				i_left = (Vs - v[0]) / sim_params.Rs
			
			# Estimate load current
			if np.isinf(sim_params.RL):
				i_right = 0.0
			else:
				i_right = v[-1] / sim_params.RL
			
			# Estimate di_dx
			di_dx = np.zeros_like(v) # Initialize as all zeros
			if Nx > 1:
				di_dx[1:-1] = (i_half[1:] - i_half[:-1]) / dx # Initialize di_dx from latest current estimates
			
			# Estimate first di_dx
			if sim_params.Rs==0:
				di_dx[0] = i_half[0] / dx
			else:
				di_dx[0] = (i_half[0] - i_left) / dx
			
			# Estimate last di_dx
			di_dx[-1] = (i_right - i_half[-1]) / dx
			
			# Update voltage from di_dx and C. Incorporate loss from G
			# v = v - dt * (di_dx / self.C_nodes) # Lossless model
			v = (v - dt * di_dx / self.C_nodes) / (1.0 + dt * self.G_nodes / self.C_nodes)

			
			# Overwrite first voltage if source impedeance is zero
			if sim_params.Rs == 0:
				v[0] = Vs
			
			# Add to history
			v_hist[n, :] = v
			i_hist[n, :] = i_half
		
		# Create time and space arrays for result
		t = np.arange(Nt) * dt
		x = np.linspace(0.0, sim_params.total_length, Nx+1)
		
		# Create result object and return
		return FiniteDiffResult(t=t, x=x, v_xt=v_hist, i_xt=i_hist)


