from dataclasses import dataclass
import numpy as np
from typing import Callable, List, Literal, Tuple, Optional

# TODO: Explain
def _newton_i_update(i0: np.ndarray, s: np.ndarray, L0: np.ndarray, alpha: np.ndarray, max_iter: int = 15, tol: float = 1e-12) -> np.ndarray:
	"""Solve per-element for i:  F(i) = i - i0 - s / (L0 * (1 + alpha*i^2)) = 0.

	Parameters
	----------
	i0 : array
		Previous current (same shape as s).
	s : array
		s = dt * Δv  (ladder)   OR   s = - dt * (∂v/∂x) (FDTD)
	L0, alpha : arrays
		Base inductance and nonlinearity per element (section or half-cell).
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

#NOTE: Previously LadderResult
@dataclass
class LumpedElementResult:
	t: np.ndarray
	v_nodes: np.ndarray    # (Nt, N+1)
	i_L: np.ndarray        # (Nt, N)
	
	#NOTE: Was probe_ladder_voltage()
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

#NOTE: Previously FDTDResult
@dataclass
class FiniteDiffResult:
	t: np.ndarray
	x: np.ndarray
	v_xt: np.ndarray   # (Nt, Nx+1)
	i_xt: np.ndarray   # (Nt, Nx)
	
	# Note: Was probe_fdtd_voltage()
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

#NOTE: Was FDTDRegion
@dataclass
class TLINRegion:
	x0: float
	x1: float
	L0_per_m: float
	C_per_m: float
	alpha: float

#NOTE: Previously LumpedElementParamsPW
@dataclass
class LumpedElementParams:
	N: int
	L: float
	Rs: float
	RL: float
	dt: float
	T: float
	Vs_func: Callable[[float], float]
	regions: List[TLINRegion]
	nonlinear_update: Literal["explicit","implicit"] = "explicit"

#NOTE: Previously FiniteDiffParamsPW
@dataclass
class FiniteDiffParams:
	Nx: int
	L: float
	dt: float
	T: float
	Rs: float
	RL: float
	Vs_func: Callable[[float], float]
	regions: List[TLINRegion]
	nonlinear_update: Literal["explicit","implicit"] = "explicit"

def _sample_regions_on_grid(regions: List, grid: np.ndarray, field: str) -> np.ndarray:
	vals = np.zeros_like(grid, dtype=float)
	for r in regions:
		mask = (grid >= r.x0) & (grid < r.x1)
		vals[mask] = getattr(r, field)
	# ensure last grid point gets last region's value
	end = max(r.x1 for r in regions)
	if np.isclose(grid[-1], end):
		for r in regions:
			if np.isclose(end, r.x1):
				vals[-1] = getattr(r, field)
	return vals

# NOTE: Previously called NLTLadderPW
class LumpedElementSim:
	''' Simulator for L-C ladder based non-linear transmission line. '''
	
	def __init__(self, p: LumpedElementParams):
		self.p = p
		self.Nt = int(np.round(p.T / p.dt)) + 1
		self.dx = p.L / p.N
		x_sec = (np.arange(p.N) + 0.5) * self.dx
		x_nodes = np.arange(p.N + 1) * self.dx
		self.L0_sec = _sample_regions_on_grid(p.regions, x_sec, 'L0_per_m') * self.dx
		self.C_nodes = _sample_regions_on_grid(p.regions, x_nodes, 'C_per_m') * self.dx
		self.alpha_sec = _sample_regions_on_grid(p.regions, x_sec, 'alpha')

	def run(self) -> LumpedElementResult:
		p = self.p
		N, dt, Nt = p.N, p.dt, self.Nt

		v = np.zeros(N+1)
		iL_half = np.zeros(N)
		v_hist = np.zeros((Nt, N+1))
		i_hist = np.zeros((Nt, N))

		for n in range(Nt):
			t_n = n * dt
			Vs = p.Vs_func(t_n)

			dv = v[:-1] - v[1:]
			if p.nonlinear_update == "explicit":
				Ld = self.L0_sec * (1.0 + self.alpha_sec * iL_half**2)
				iL_half = iL_half + dt * dv / Ld
			else:
				s = dt * dv
				iL_half = _newton_i_update(iL_half, s, self.L0_sec, self.alpha_sec)

			# Node updates
			dvdt = np.zeros_like(v)
			if p.Rs == 0:
				v[0] = Vs
			else:
				i_src = 0.0 if np.isinf(p.Rs) else (Vs - v[0]) / p.Rs
				dvdt[0] = (i_src - iL_half[0]) / self.C_nodes[0]

			if N > 1:
				dvdt[1:-1] = (iL_half[:-1] - iL_half[1:]) / self.C_nodes[1:-1]

			i_load = 0.0 if np.isinf(p.RL) else v[-1] / p.RL
			dvdt[-1] = (iL_half[-1] - i_load) / self.C_nodes[-1]

			v = v + dt * dvdt

			v_hist[n, :] = v
			i_hist[n, :] = iL_half

		t = np.arange(Nt) * dt
		return LumpedElementResult(t=t, v_nodes=v_hist, i_L=i_hist)


#NOTE: Previously NLTFDTD_PW
class FiniteDiffSim:
	def __init__(self, p: FiniteDiffParams):
		self.p = p
		self.dx = p.L / p.Nx
		self.Nt = int(np.round(p.T / p.dt)) + 1
		x_nodes = np.linspace(0.0, p.L, p.Nx + 1)
		x_half  = (np.arange(p.Nx) + 0.5) * self.dx
		self.C_nodes = _sample_regions_on_grid(p.regions, x_nodes, 'C_per_m')
		self.L0_half = _sample_regions_on_grid(p.regions, x_half,  'L0_per_m')
		self.alpha_half = _sample_regions_on_grid(p.regions, x_half, 'alpha')

	@staticmethod
	def cfl_dt(dx: float, Lmin: float, Cmin: float, safety: float = 0.9) -> float:
		v = 1.0 / np.sqrt(max(Lmin, 1e-300) * max(Cmin, 1e-300))
		return safety * dx / v

	def run(self) -> FiniteDiffResult:
		p = self.p
		Nx, dt, Nt = p.Nx, p.dt, self.Nt
		dx = self.dx

		v = np.zeros(Nx+1)
		i_half = np.zeros(Nx)
		v_hist = np.zeros((Nt, Nx+1))
		i_hist = np.zeros((Nt, Nx))

		for n in range(Nt):
			t_n = n * dt
			Vs = p.Vs_func(t_n)

			dv_dx = (v[1:] - v[:-1]) / dx
			if p.nonlinear_update == "explicit":
				Ld_half = self.L0_half * (1.0 + self.alpha_half * i_half**2)
				i_half = i_half - dt * dv_dx / Ld_half
			else:
				s = - dt * dv_dx
				i_half = _newton_i_update(i_half, s, self.L0_half, self.alpha_half)

			# boundaries
			if p.Rs == 0:
				i_left = None
			else:
				i_left = (Vs - v[0]) / p.Rs
			i_right = 0.0 if np.isinf(p.RL) else v[-1] / p.RL

			di_dx = np.zeros_like(v)
			if Nx > 1:
				di_dx[1:-1] = (i_half[1:] - i_half[:-1]) / dx
			di_dx[0]  = (i_half[0] - (0.0 if p.Rs==0 else i_left)) / dx
			di_dx[-1] = (i_right - i_half[-1]) / dx

			v = v - dt * (di_dx / self.C_nodes)

			if p.Rs == 0:
				v[0] = Vs

			v_hist[n, :] = v
			i_hist[n, :] = i_half

		t = np.arange(Nt) * dt
		x = np.linspace(0.0, p.L, Nx+1)
		return FiniteDiffResult(t=t, x=x, v_xt=v_hist, i_xt=i_hist)


