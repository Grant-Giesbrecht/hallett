from dataclasses import dataclass, field
import numpy as np
from typing import Literal, Tuple, Optional
import matplotlib.pyplot as plt

from stardust.serializer import Serializable

from hallett.nltsim.core import FiniteDiffResult
from hallett.core import W_to_dBm

def analytic_signal(x: np.ndarray) -> np.ndarray:
	''' Creates an analytic signal from a real-valued signal: runs an FFT, removes all negative
	components and doubles positive components, runs an inverse fourier transform to return to
	time domain.
	'''
	
	x = np.asarray(x)
	N = x.size
	Xf = np.fft.fft(x, n=N)
	
	# Create single sided spectrum w mask `H` from double sided spectrum
	H = np.zeros(N, dtype=float)
	if N % 2 == 0:
		H[0] = 1.0
		H[N//2] = 1.0
		H[1:N//2] = 2.0
	else:
		H[0] = 1.0
		H[1:(N+1)//2] = 2.0
	Zf = Xf * H
	
	# Recreate time domain signal from inverse fourier transform - will be complex valued
	z = np.fft.ifft(Zf, n=N)
	return z

@dataclass
class HilbertResult:
	''' Dataclass to save results from `hilbert_phase_diagnostic` function.
	'''
	
	envelope: np.ndarray
	phase: np.ndarray
	inst_freq_hz: np.ndarray
	analytic: np.ndarray
	t: Optional[np.ndarray] = None

def hilbert_phase_diagnostic(v: np.ndarray, dt: float, center_freq_hz: Optional[float] = None, smooth_points: int = 0) -> HilbertResult:
	''' Calculates the envelope, phase, and inst. frequency from a real-valued signal. NOTE: The inst. frequency
	is based solely on ∆phase/∆t, so smoothing (set smoothing_points > 1) is recommended. 
	'''
	
	# Ensure v is a numpy array
	v = np.asarray(v, dtype=float)
	
	# Get the analytic version of the real valued signal v
	z = analytic_signal(v)
	
	# Envelope is the magnitude of the complex-valued signal
	env = np.abs(z)
	
	# Get phase from complex 
	phase = np.unwrap(np.angle(z))
	
	# discretized inst. frequency from dt
	inst_omega = np.gradient(phase, dt)
	
	# Convert to frequency
	inst_freq = inst_omega / (2.0 * np.pi)
	
	# Perform smoothing
	if smooth_points and smooth_points > 1:
		k = int(max(1, smooth_points))
		w = np.ones(k, dtype=float) / k
		inst_freq = np.convolve(inst_freq, w, mode='same')
	
	# Shift frequency as offset from center if requested
	if center_freq_hz is not None:
		inst_freq = inst_freq - float(center_freq_hz)
	
	# Create time list
	t = np.arange(v.size) * dt
	
	# Save results in dataclass
	return HilbertResult(envelope=env, phase=phase, inst_freq_hz=inst_freq, analytic=z, t=t)

@dataclass
class SpectrumResult:
	''' Dataclass to save results from `spectrum_probe`. '''
	
	freqs_hz: np.ndarray
	spec: np.ndarray
	scaling: Literal["magnitude","power","psd"]

def spectrum_probe(v: np.ndarray, dt: float, window: Literal["hann","rect"] = "hann", nfft: Optional[int] = None, onesided: bool = True, detrend: bool = True, scaling: Literal["magnitude","power","psd"] = "psd") -> SpectrumResult:
	''' Calculates the spectrum of a signal.
	
	Parameters:
		v (list): List of amplitude values
		dt (list): List of time values
		detrend (bool): Removes DC offset from input signal. Default = True
		nfft (int): Number of points to inlcude in FFT. If longer, will pad. Else will trim. (Defaults to length of series)
		window (str): `hann` or `rect`. Select window type for FFT. Default to `hann`.
		scaling (str): `magnitude`, `power`, or `psd`. Selects output scaling type. Defaults to `psd`.
		onesided (bool): Selects one vs two-sided FFT. Default is True.
		
	Returns:
		SpectrumResult: Dataclass containing the results.
	'''
	
	# Ensure input is a numpy array
	x = np.asarray(v, dtype=float)
	
	# Remove DC offset if requested
	if detrend:
		x = x - np.mean(x)
		
	# Get FFT length
	N = x.size
	if nfft is None:
		nfft = N
	
	# Select a windowing method
	if window == "hann":
		w = np.hanning(N)
	elif window == "rect":
		w = np.ones(N)
	else:
		raise ValueError("Unsupported window: %r" % window)
	
	# Apply window
	xw = x * w
	
	# Trim/lengthen time series until length == nfft
	if nfft != N:
		if nfft > N:
			xw = np.pad(xw, (0, nfft - N))
			w = np.pad(w, (0, nfft - N))
		else:
			xw = xw[:nfft]
			w = w[:nfft]
	
	# Get sample rate
	Fs = 1.0 / dt
	
	# Compute single or double ended FFT
	if onesided:
		V = np.fft.rfft(xw)
		freqs = np.fft.rfftfreq(nfft, d=dt) # real-valued fft
	else:
		V = np.fft.fft(xw)
		freqs = np.fft.fftfreq(nfft, d=dt) # complex(?) valued fft
	
	# Select signal output scaling
	if scaling == "magnitude":
		spec = np.abs(V)
	elif scaling == "power":
		spec = np.abs(V)**2
	elif scaling == "psd":
		U = (w[:N]**2).mean() if nfft == N else (w**2).mean()
		spec = (np.abs(V)**2) / (Fs * nfft * U)
	else:
		raise ValueError("Unsupported scaling: %r" % scaling)
	
	
	return SpectrumResult(freqs_hz=freqs, spec=spec, scaling=scaling)


def time_gate(t: np.ndarray, x: np.ndarray, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
	''' Masks a time list s.t. it returns only the waveform at times between t0
	and t1, and returns 0 for everything else. 
	'''
	
	t = np.asarray(t)
	x = np.asarray(x)
	m = (t >= t0) & (t <= t1)
	y = np.zeros_like(x)
	y[m] = x[m]
	return t, y

def plot_signal_diagnostics(t, v, dt, center_freq_hz=None, smooth_points=0, fmax=None):
	"""
	Make a 3-panel diagnostic plot:
	1. Time-domain waveform and envelope
	2. Instantaneous frequency (chirp)
	3. Spectrum (log scale)
	"""
	h = hilbert_phase_diagnostic(v, dt, center_freq_hz=center_freq_hz, smooth_points=smooth_points)
	spec = spectrum_probe(v, dt, window="hann", scaling="psd")

	fig, axes = plt.subplots(3, 1, figsize=(10, 8))
	ax1, ax2, ax3 = axes

	# Panel 1: waveform + envelope
	ax1.plot(t*1e9, v, label="waveform")
	ax1.plot(h.t*1e9, h.envelope, label="envelope", lw=2)
	ax1.set_xlabel("Time (ns)")
	ax1.set_ylabel("Voltage (V)")
	ax1.legend()
	ax1.set_title("Waveform and envelope")

	# Panel 2: instantaneous frequency
	ax2.plot(h.t*1e9, h.inst_freq_hz/1e9)
	ax2.set_xlabel("Time (ns)")
	ax2.set_ylabel("Inst. freq (GHz)")
	ax2.set_title("Instantaneous frequency (chirp)")

	# Panel 3: spectrum
	ax3.semilogy(spec.freqs_hz/1e9, spec.spec)
	ax3.set_xlabel("Frequency (GHz)")
	ax3.set_ylabel(spec.scaling)
	ax3.set_title("Spectrum")
	if fmax is not None:
		ax3.set_xlim(0, fmax/1e9)

	plt.tight_layout()
	return fig, axes

def tone_power_from_psd(freqs: np.ndarray, psd: np.ndarray, f_target: float, bw_bins: int, RL: float = 50.0) -> float:
	"""
	Estimate power at f_target (fundamental/harmonics) by integrating PSD over +/- bw_bins bins.
	PSD has units ~ V^2/Hz. Multiply by bin bandwidth (df) and sum to get V^2, then P=V_rms^2/RL.
	
	Params:
		freqs (list): List of frequency values for psd list. Assumes equal spacing.
		psd (list): Power spectral density list, in V^2/Hz #TODO: convert to dBm/Hz? or W/sqrt(Hz)?
		f_target (float): Center freqeuncy around which to integrate
		bw_bins (int); Bandwidth in bins over which to integrate.
		RL (float): Load impedance from which to convert voltage to power.
		
	Returns:
		float: Returns pwoer in watts 
	"""
	
	# Ensure enough points exist
	if len(freqs) < 3:
		return np.nan
	
	# Get ∆f
	df = freqs[1] - freqs[0]
	
	# Get index of smallest frequency delta
	k0 = int(np.argmin(np.abs(freqs - f_target)))
	
	# Get indices above and below
	k_lo = max(0, k0 - bw_bins)
	k_hi = min(len(freqs)-1, k0 + bw_bins)
	
	# Integrate with trapezoid method over region
	v2 = np.trapz(psd[k_lo:k_hi+1], dx=df)  # integrate PSD over band -> V^2
	
	# Convert to power
	P = v2 / RL                             # watts (since V_rms^2/R)
	
	# Return power in region of spectrum
	return P

@dataclass
class HarmonicPowerResult(Serializable):
	
	__state_fields__ = ("x_parameter", "x_values", "f0", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "sim_ids" )
	
	def __init__(self, x_parameter:str="", x_values:list=[]):
		super().__init__()
		
		self.x_parameter = x_parameter
		self.x_values = x_values
		
		self.f0 = []
		self.h2 = []
		self.h3 = []
		self.h4 = []
		self.h5 = []
		self.h6 = []
		self.h7 = []
		self.h8 = []
		
		self.sim_ids = []
	 
	def append_point(self, harmonics:list, x_value:float=None, sim_id:str=None):
		''' Accepts a list of harmonic powers, and appends them to the lists, also
		appending the specified x-value.'''
		
		if x_value is not None:
			self.x_values.append(x_value)
		
		if sim_id is not None:
			self.sim_ids.append(sim_id)
		
		for idx, val in enumerate(harmonics):
			self.append_harmonic(idx+1, val)
	
	def append_harmonic(self, harm_idx:int, value:float):
		''' Appends the specified value to the specified harmonic list. Adding a 
		single data point to a specific harmonic.
		'''
		
		if harm_idx == 1:
			self.f0.append(value)
		elif harm_idx == 2:
			self.h2.append(value)
		elif harm_idx == 3:
			self.h3.append(value)
		elif harm_idx == 4:
			self.h4.append(value)
		elif harm_idx == 5:
			self.h5.append(value)
		elif harm_idx == 6:
			self.h6.append(value)
		elif harm_idx == 7:
			self.h7.append(value)
		elif harm_idx == 8:
			self.h8.append(value)
	
def load_harmonics_probe(result_obj, f0:float, tail:float, bin_bw_Hz:float, num_harmonics:int=3, RL:float=50, x_parameter:str="", x_values:list=[]):
	'''
	Calculates the power at a specified number of harmonics at the system load
	from a result object. Returns a list of the harmonic powers.
	
	Args:
		result_obj: Simulation result object to use.
		tail (float): Length in seconds of time series, starting at the end, to 
			analyze. This allows you to skip the beginning of the sequence when
			the system might not be in steady state.
	'''
	
	# Get v and t from reulst object
	if isinstance(result_obj, FiniteDiffResult):
		v_t = result_obj.v_xt[:, -1]
	else:
		v_t = result_obj.v_nodes[:, -1]
	t = result_obj.t

	# Tail for steady-state
	if tail is not None and tail > 0:
		t0 = max(0.0, t[-1] - tail)
		mask = t >= t0
		
		# Trim time and voltage to specified tail
		t = t[mask]
		v_t = v_t[mask]
	
	dt = t[1] - t[0]
	spec = spectrum_probe(v_t, dt, window="hann", scaling="psd")
	f = spec.freqs_hz
	psd = spec.spec
	
	output_elements = []
	for i in range(1, num_harmonics+1):
		harm_power = tone_power_from_psd(f, psd, i*f0, bin_bw_Hz, RL=RL)
		harm_power_dBm = W_to_dBm(harm_power)
		
		# Build up output list
		output_elements.append(harm_power_dBm)
		
		# # Add to HP object if present
		# if hp_obj is not None:
		# 	hp_obj.append_harmonic(i, harm_power_dBm)	
	
	# print(f"Vdc={vb:.3f} m  ->  P1={P1_dBm:.2f} dBm,  P2={P2_dBm:.2f} dBm,  P3={P3_dBm:.2f} dBm")
	return output_elements