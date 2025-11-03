import numpy as np

def lin_to_dB(x_lin:float, use10:bool=False) -> float:
	''' Converts a linear parameter to decibels. Will raise a warning if
	negative numbers are provided.
	
	Args:
		x_lin (float): Linear value to convert to decibels
		use10 (bool): Use 10*log(X) definition instead of 20*log(X) definition. Default is false.
	
	Returns:
		(float): Value converted to dB
	'''
	
	if use10:
		return 10*np.log10(x_lin)
	else:
		return 20*np.log10(x_lin)

def dB_to_lin(x_dB:float, use10:bool=False) -> float:
	''' Converts a linear parameter to decibels. Will raise a warning if
	negative numbers are provided.
	
	Args:
		x_dB (float): Value in dB to convert to linear units.
		use10 (bool): Use 10*log(X) definition instead of 20*log(X) definition. Default is false.
	
	Returns:
		(float): Value converted to dB
	'''
	
	if use10:
		return np.log10(10, x_dB/10)
	else:
		return np.power(10, x_dB/20)

def W_to_dBm(Pw):
	return 10*np.log10(Pw/1e-3) if Pw > 0 else -np.inf