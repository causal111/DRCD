# -*- coding: utf-8 -*-

from other_methods.MIC import CausalityDetector as MIC
from lingam import LiM
from other_methods import HCM
from other_methods import GSF
from other_methods import crack_cpp
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
import sys, os
from ucimlrepo import fetch_ucirepo
import os
from typing import List, Tuple, Dict
import scripts.generate_bipartite_graph as generate_bipartite_graph
import DRCD



def prepare_datasets(X, y) -> List[Tuple[str, np.ndarray]]:
	"""
	Prepare datasets for causal analysis.
	
	Parameters
	----------
	X : pandas.DataFrame
		Feature data from the UCI heart disease dataset.
	y : pandas.DataFrame
		Target data from the UCI heart disease dataset.
		
	Returns
	-------
	List[Tuple[str, np.ndarray]]
		A list of tuples where each tuple contains:
		- Dataset name (string)
		- Data array (shape: n_samples x 2)
	"""

	datasets = []
	
	# Define feature and target combinations
	feature_columns = ["oldpeak", "trestbps", "chol", "thalach", "age"]
	target_mappings = [
		("num", lambda df: y.to_numpy().T[0]),
		("sex", lambda df: df["sex"]),
		("cp", lambda df: df["cp"]),
		("fbs", lambda df: df["fbs"]),
		("restecg", lambda df: df["restecg"]),
		("exang", lambda df: df["exang"]),
		("slope", lambda df: df["slope"])
	]
	
	# Create datasets for all combinations
	for feature in feature_columns:
		for target_name, target_func in target_mappings:
			data = np.zeros((len(X), 2))
			data[:, 0] = X[feature].to_numpy()
			data[:, 1] = target_func(X)
			
			dataset_name = f"{feature}_vs_{target_name}"
			datasets.append((dataset_name, data))
	
	return datasets


def analyze_causality(data: np.ndarray) -> Dict[str, int]:
	"""
	Analyze causal relationships using multiple methods.
	
	Parameters
	----------
	data : np.ndarray
		Input data array with shape (n_samples, 2).
		First column is the continuous variable 'X'.
		Second column is the discrete variable 'Y'.
		
	Returns
	-------
	Dict[str, int]
		Dictionary of results from each method:
		- 'drcd': DRCD method result
		- 'lim': LiNGAM method result
		- 'mic': MIC method result
		- 'crack': CRACK method result
		- 'hcm': HCM method result
		- 'gsf': GSF method result
		
		For each method, the result is an integer:
		0 = No causality detected
		1 = X causes Y (X→Y)
		2 = Y causes X (Y→X)
	"""


	results = {}
	
	# Analysis using DRCD
	drcd = DRCD.DRCD(data)
	results['drcd'] = drcd.infer()
	
	# Analysis using LiM
	model = LiM()
	with redirect_stdout(open(os.devnull, 'w')):
		model.fit(data, np.array([[1,0]]))
	
	if not model._adjacency_matrix[0][1] == 0:
		results['lim'] = 1
	elif not model._adjacency_matrix[1][0] == 0:
		results['lim'] = 2
	else:
		results['lim'] = 0
	
	# Analysis using MIC
	detector = MIC()
	results['mic'] = detector.detect_causality(data[:,0], data[:,1])

	# Analysis using CRACK
	results['crack'] = crack_cpp.run_crack(data[:,0], data[:,1], os.getcwd()+"/other_methods/crack/code", threshold=0.01)

	# Analysis using HCM
	results['hcm'] = HCM.execute(data)

	# Analysis using GSF
	results['gsf'] = GSF.infer_causal_direction(data)
	
	return results




def experiment_with_real_data():
	"""
	Run causal discovery experiments using the UCI heart disease dataset.
	
	This function:
	1. Retrieves the UCI heart disease dataset
	2. Prepares multiple datasets by combining different features with target variables
	3. Applies various causal discovery methods to each dataset
	4. Compiles and saves the results
	5. Generates visualization of the results
	
	Results are saved as a CSV file and as a bipartite graph visualization.
	"""
	
	# Retrieve dataset
	heart_disease = fetch_ucirepo(id=45)
	X = heart_disease.data.features
	y = heart_disease.data.targets
	
	# Prepare all datasets
	datasets = prepare_datasets(X, y)
	
	# DataFrame to store results
	results_df = pd.DataFrame(columns=['Dataset', 'DRCD', 'LiM', 'MIC', 'CRACK'])
	
	# Run analysis for each dataset
	for dataset_name, data in datasets:
		print(dataset_name)
		results = analyze_causality(data)

		# Add results to the DataFrame
		new_row = pd.DataFrame([{
			'Dataset': dataset_name,
			'DRCD': results['drcd'],
			'LiM': results['lim'],
			'MIC': results['mic'],
			'CRACK': results['crack'],
			'HCM': results['hcm'],
			'GSF': results['gsf']
		}])
		results_df = pd.concat([results_df, new_row], ignore_index=True)
	
	# Display results
	print("=== Causality Analysis Results ===")
	print(results_df)
	
	# Create directory to save results
	results_dir = 'results/real'
	os.makedirs(results_dir, exist_ok=True)

	# Save results as CSV
	results_df.to_csv('./results/real/causality_analysis_results.csv', index=False)
	
	# Generate and save visualizations
	generate_bipartite_graph.main('./results/real/causality_analysis_results.csv', './results/real/bipartite_graphs.svg')



if __name__ == '__main__':
	experiment_with_real_data()
	

