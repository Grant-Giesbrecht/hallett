import numpy as np

# ----------------- Waveform Helpers ----------------------------

def gaussian_pulse(t:float, t0:float, sigma:float, V0:float) -> float:
	return V0 * np.exp(-0.5*((t - t0)/sigma)**2)

def raised_cosine_step(t:float, t0:float, rise:float, V1:float=1) -> float:
	''' Function for helping to create input stimuli. 
	
	Parameters:
		t (float): Time point to evaluate
		t0 (float): Center of ramp up time
		rise (float): Rise time
		v1 (float): High voltage. 
	'''
	
	if t < t0 - 0.5*rise:
		return 0.0
	if t > t0 + 0.5*rise:
		return V1
	phase = (t - (t0 - 0.5*rise)) / rise * np.pi
	return 0.5*V1*(1 - np.cos(phase))