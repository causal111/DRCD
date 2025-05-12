# -*- coding: utf-8 -*-

import subprocess
import pandas as pd
import os
import numpy as np


def parse_crack_result(result_text):
	"""
	Extract main results from Crack's output
	
	Returns:
		dict: {
			'causal_direction': 'X->Y' or 'Y->X' or 'X--Y',
			'epsilon': float,
			'delta_x_to_y': float,
			'delta_y_to_x': float,
			'runtime': float
		}
	"""
	lines = result_text.split('\n')
	result = {}
	
	for line in lines:
		if line.startswith('deltaX->Y:'):
			result['delta_x_to_y'] = float(line.split(':')[1].strip())
		elif line.startswith('deltaY->X:'):
			result['delta_y_to_x'] = float(line.split(':')[1].strip())
		elif line.startswith('Result:'):
			parts = line.split()
			result['causal_direction'] = parts[1]
			result['epsilon'] = float(parts[-1])
		elif line.startswith('Runtime:'):
			result['runtime'] = float(line.split(':')[1].split()[0])
	
	return result




def create_binary_test_data(X, Y, test_package_dir):
	
	# Create DataFrame
	data = pd.DataFrame({
		'X': X,
		'Y': Y
	})

	test_file = os.path.join(test_package_dir, "test_data.txt")
	
	# Save data with space delimiter
	data.to_csv(test_file, sep=' ', index=False, header=False)
	
	return test_file


def run_crack(X, Y, test_package_dir, threshold = None, debug=False):

	test_file = create_binary_test_data(X, Y, test_package_dir)

	try:
		crack_path = os.path.join(test_package_dir, "crack.run")
		results_dir = os.path.join(test_package_dir, "results")
		
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
		
		cmd = [
			crack_path,
			"-s", "1",
			"-o", f"{results_dir}/test_",
			"-x", "1",
			"-c",
			"-d", " ",
			"-i", test_file,
			"-a", "test",
			"-t", "n"  # X is numeric type
		]
		
		# print("\nCommand to execute:", ' '.join(cmd))
		
		result = subprocess.run(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			cwd=test_package_dir
		)
		
		if result.returncode != 0:
			raise Exception(f"Crack execution error:\nstdout: {result.stdout}\nstderr: {result.stderr}")
			
		# print("\nStandard output:", result.stdout)
		# print("Standard error:", result.stderr)
			
		# return result.stdout
		
	except Exception as e:
		print(f"An error occurred: {str(e)}")
		return None

	result = parse_crack_result(result.stdout)

	if debug:
		print(result)


	if (threshold is not None) and (result['epsilon'] < threshold):
		return 0

	if result['causal_direction'] == "X--Y":
		return 0
	elif result['causal_direction'] == "X->Y":
		return 1
	else:
		return 2