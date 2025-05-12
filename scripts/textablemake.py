# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import pandas as pd
from scipy.stats import binom

def generate_latex_table(results_file='results/artificial/all_results.pkl', include_overall=True, DRCDonly=False):
	"""
	Reads the pickle file with experiment results and generates LaTeX tables
	using booktabs style with methods as rows and causal types as columns
	
	Parameters:
	-----------
	results_file : str
		Path to the pickle file containing experiment results
	include_overall : bool
		Whether to include an Overall column (across all causal types) for each method
	"""
	
	# Load the results
	with open(results_file, 'rb') as f:
		all_results = pickle.load(f)
	
	# Extract methods
	methods = list(all_results.keys())
	
	# Create accuracy tables
	print("Generating booktabs LaTeX tables...")
	
	causal_types = ['No Causation', 'X->Y', 'Y->X']
	
	# Add Overall to column headers if requested
	display_columns = causal_types.copy()
	if include_overall:
		display_columns.append('Overall')
	
	# Replace underscores in method names
	formatted_methods = []
	for method in methods:
		# Replace underscores with escaped underscores
		if '_' in method:
			formatted_method = method.replace('_', '\\_')
		else:
			formatted_method = method
		formatted_methods.append(formatted_method)
	
	# Create table header
	latex_table = "\\begin{table}[htbp]\n"
	latex_table += "\\centering\n"
	latex_table += "\\caption{Accuracy of causal discovery methods with 95\\% confidence intervals}\n"
	latex_table += "\\begin{tabular}{l" + "c" * len(display_columns) + "}\n"
	latex_table += "\\toprule\n"
	
	# Column headers
	latex_table += "Method & " + " & ".join(display_columns) + " \\\\\n"
	latex_table += "\\midrule\n"
	
	# For each method
	for i, method in enumerate(methods):
		formatted_method = formatted_methods[i]
		latex_row = formatted_method + " & "
		
		# For each causal type
		for j, causal_type in enumerate(causal_types):
			result_key = f"For {causal_type} data"
			if result_key in all_results[method]:
				accuracy = all_results[method][result_key]['Correct prediction rate']
				n_samples = all_results[method][result_key]['Sample count']
				
				# Calculate confidence interval
				lower, upper = calculate_confidence_interval(accuracy, n_samples)
				
				# Format as percentage with confidence interval
				cell = f"{accuracy*100:.1f}\\% ({lower*100:.1f}--{upper*100:.1f}\\%)"
				latex_row += cell
			else:
				latex_row += "N/A"
				
			if j < len(causal_types) - 1 or include_overall:
				latex_row += " & "
		
		# Add Overall column if requested
		if include_overall:
			# Get the overall performance from the results
			if 'Overall Performance' in all_results[method]:
				overall_accuracy = all_results[method]['Overall Performance']['Accuracy']
				total_samples = all_results[method]['Overall Performance']['Total samples']
				
				# Calculate confidence interval
				lower, upper = calculate_confidence_interval(overall_accuracy, total_samples)
				
				# Format as percentage with confidence interval
				cell = f"{overall_accuracy*100:.1f}\\% ({lower*100:.1f}--{upper*100:.1f}\\%)"
				latex_row += cell
			else:
				latex_row += "N/A"
		
		latex_row += " \\\\\n"
		latex_table += latex_row
	
	# Close the table
	latex_table += "\\bottomrule\n"
	latex_table += "\\end{tabular}\n"
	latex_table += "\\label{tab:accuracy_results}\n"
	latex_table += "\\end{table}\n"
	
	# Save to file
	results_dir = 'results/artificial'
	os.makedirs(results_dir, exist_ok=True)
	
	# Add suffix if table includes overall column
	filename = 'accuracy_table.tex'
	if include_overall:
		filename = 'accuracy_table_with_overall.tex'

	if DRCDonly:
		filename = 'accuracy_table_with_overall_DRCDonly.tex'
	
	with open(os.path.join(results_dir, filename), 'w') as f:
		f.write(latex_table)
	
	print(f"Booktabs LaTeX table saved to {os.path.join(results_dir, filename)}")


def calculate_confidence_interval(accuracy, n_samples, confidence=0.95):
	"""
	Calculate binomial confidence interval for accuracy
	
	Parameters:
	-----------
	accuracy : float
		Accuracy rate (proportion of correct predictions)
	n_samples : int
		Number of samples
	confidence : float
		Confidence level (default: 0.95 for 95% CI)
		
	Returns:
	--------
	tuple
		(lower_bound, upper_bound) of the confidence interval
	"""
	n_correct = int(accuracy * n_samples)
	alpha = 1 - confidence
	
	# Use binomial distribution to calculate confidence interval
	lower = binom.ppf(alpha/2, n_samples, accuracy) / n_samples
	upper = binom.ppf(1 - alpha/2, n_samples, accuracy) / n_samples
	
	# Handle edge cases
	if np.isnan(lower):
		lower = 0.0
	if np.isnan(upper):
		upper = 1.0
	
	return (lower, upper)
