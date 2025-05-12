# -*- coding: utf-8 -*-

import numpy as np
import os
import matlab.engine

def infer_causal_direction(data: np.ndarray, matlab_dir: str = './other_methods/GES/GES_with_generalized_score') -> int:
	"""
	Analyze causal relationships between a continuous variable (1st column) and a discrete variable (2nd column)
	to determine the direction of causality.
	
	Parameters:
	-----------
	data : np.ndarray
		A NumPy array with 2 columns. Assumes 1st column is continuous and 2nd column is discrete.
	matlab_dir : str
		Path to the directory containing the MATLAB code (GES algorithm)
		
	Returns:
	--------
	int
		0: No causal relationship (independent)
		1: Continuous variable causes discrete variable
		2: Discrete variable causes continuous variable
	"""
	if data.shape[1] != 2:
		raise ValueError("Data must have exactly 2 columns (2 variables)")
	
	# Convert discrete variable values to integers
	processed_data = data.copy()
	
	# Convert 2nd column (discrete variable) to integer values
	unique_vals = np.unique(data[:, 1])
	mapping = {val: i for i, val in enumerate(unique_vals)}
	processed_data[:, 1] = np.array([mapping[val] for val in data[:, 1]])
	
	# Normalize 1st column (continuous variable)
	processed_data[:, 0] = (data[:, 0] - np.mean(data[:, 0])) / np.std(data[:, 0])
	
	# Start MATLAB engine
	eng = matlab.engine.start_matlab()
	
	try:
		# Navigate to the MATLAB code directory
		eng.cd(matlab_dir)
		
		# Add all subdirectories to path
		p = eng.genpath(eng.pwd())
		eng.addpath(p, nargout=0)
		
		# Convert Python array to MATLAB array
		matlab_X = matlab.double(processed_data.tolist())
		
		# Set GES parameters
		parameters = {'kfold': 10.0, 'lambda': 0.01}
		maxP = 1.0  # Maximum number of parents (set to 1 for 2 variables)
		multi_sign = 0.0  # Not multi-dimensional variables
		
		# Execute GES function
		record = eng.GES(matlab_X, 1.0, multi_sign, maxP, parameters, nargout=1)
		
		# Interpret results
		G = np.array(record['G'])
		
		# Check strength of correlation
		correlation = np.abs(np.corrcoef(data.T)[0, 1])
		
		# Determine direction based on causal matrix
		if np.all(G == 0) or correlation < 0.1:
			# No causal relationship, or very weak correlation
			return 0
		elif G[0, 1] == 1:
			# Continuous -> Discrete
			return 1
		elif G[1, 0] == 1:
			# Discrete -> Continuous
			return 2
		elif G[0, 1] == -1 or G[1, 0] == -1:
			# Undetermined case, additional analysis needed
			
			# Calculate variance of continuous values for each discrete value
			discrete_vals = np.unique(data[:, 1])
			variances = []
			group_sizes = []
			
			for val in discrete_vals:
				subset = data[data[:, 1] == val, 0]
				if len(subset) > 1:
					variances.append(np.var(subset))
					group_sizes.append(len(subset))
			
			if not variances:
				# Insufficient data
				return 0
			
			# Consider variance of each group
			weighted_mean_var = sum(v * s for v, s in zip(variances, group_sizes)) / sum(group_sizes)
			total_var = np.var(data[:, 0])
			
			# If variance of continuous values for each discrete value is small,
			# discrete->continuous direction is more likely.
			# If variance is close to total variance, continuous->discrete is more likely.
			if weighted_mean_var < 0.7 * total_var:
				return 2  # Discrete -> Continuous
			else:
				return 1  # Continuous -> Discrete
		else:
			# Unexpected matrix case
			return 0
	
	finally:
		# Terminate MATLAB engine
		eng.quit()
