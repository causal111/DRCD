# -*- coding: utf-8 -*-

# from other_methods.MIC import CausalityDetector as MIC
# from lingam import LiM
# from other_methods import HCM
# from other_methods import GSF
# from other_methods import crack_cpp
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
import sys, os
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom
import scripts.textablemake as textablemake
from tqdm import tqdm
import DRCD


class MixedDistribution:
	def __init__(self, size):
		# Randomly determine mixing ratios
		self.weights = np.random.dirichlet(np.ones(3))
		self.size = size
		
		# Randomly set parameters for each distribution
		self.normal_params = {
			'loc': np.random.uniform(-3, 3),
			'scale': np.random.uniform(0.5, 3)
		}
		self.t_params = {
			'df': 1
		}
		self.laplace_params = {
			'loc': np.random.uniform(-3, 3),
			'scale': np.random.uniform(0.5, 3)
		}
	
	def sample(self):
		# Determine number of samples from each distribution
		nums = np.random.multinomial(self.size, self.weights)
		
		# Sample from each distribution
		normal_samples = np.random.normal(
			self.normal_params['loc'],
			self.normal_params['scale'],
			nums[0]
		)
		t_samples = stats.t.rvs(
			df=self.t_params['df'],
			size=nums[1]
		)
		laplace_samples = np.random.laplace(
			self.laplace_params['loc'],
			self.laplace_params['scale'],
			nums[2]
		)
		
		# Combine and shuffle samples
		samples = np.concatenate([normal_samples, t_samples, laplace_samples])
		np.random.shuffle(samples)
		return samples


def generate_dataset(n_samples=1000, n_datasets_per_type=1000):
	"""
	Function to generate simulation datasets
	
	Parameters:
	-----------
	n_samples : int
		Number of samples in each dataset
	n_datasets_per_type : int
		Number of datasets per causal pattern
	
	Returns:
	--------
	list of tuples
		Each tuple contains (data, causal_type)
		data is a numpy array of shape (n_samples, 2)
		causal_type is 0: none, 1: X->Y, 2: Y->X
	"""
	datasets = []

	np.random.seed(1)
	
	# Generate datasets for each causal pattern
	for causal_type in [0, 1, 2]:  # 0: none, 1: X->Y, 2: Y->X
		for _ in range(n_datasets_per_type):
			if causal_type == 0:  # No causal relationship
				X = MixedDistribution(n_samples).sample()
				Y = (MixedDistribution(n_samples).sample() > 0).astype(int)
				
			elif causal_type == 1:  # X->Y
				# Generate X from uniform distribution
				X = np.random.uniform(-3, 3, n_samples)
				# Generate coefficient with absolute value between 0.5 and 2
				a = np.random.uniform(0.5, 2)
				if np.random.random() < 0.5:  # 50% chance of negative coefficient
					a = -a
				b = np.random.uniform(-1, 1)
				# Generate noise from mixed distribution
				# noise = np.random.logistic(0, 1, n_samples)
				noise = MixedDistribution(n_samples).sample()
				# noise = np.random.normal(0, 1, n_samples)
				Y = (a * X + noise > b).astype(int)
				
			else:  # Y->X (causal_type == 2)
				Y = (MixedDistribution(n_samples).sample() > 0).astype(int)
				X = np.zeros(n_samples)
				
				# Use different distributions for Y=1 and Y=0
				dist1 = MixedDistribution(n_samples)
				dist2 = MixedDistribution(n_samples)
				
				Y1_indices = np.where(Y == 1)[0]
				Y0_indices = np.where(Y == 0)[0]
				
				X[Y1_indices] = dist1.sample()[:len(Y1_indices)]
				X[Y0_indices] = dist2.sample()[:len(Y0_indices)]
			
			# Combine data into shape (n_samples, 2)
			data = np.column_stack((X, Y))
			datasets.append((data, causal_type))
	
	return datasets


def evaluate_method(true_relations, predicted_relations):
	"""
	Function to evaluate the performance of causal discovery methods
	
	Parameters:
	-----------
	true_relations : list
		List of true causal relationships (0: none, 1: X->Y, 2: Y->X)
	predicted_relations : list
		List of predicted causal relationships (0: none, 1: X->Y, 2: Y->X)
	
	Returns:
	--------
	dict
		Performance metrics for each causal pattern and overall performance
	"""
	true_relations = np.array(true_relations)
	predicted_relations = np.array(predicted_relations)
	
	# Calculate confusion matrix
	conf_matrix = confusion_matrix(true_relations, predicted_relations)
	
	# Dictionary to store results for each causal pattern
	results = {}
	
	# Evaluate for each causal pattern (none, X->Y, Y->X)
	patterns = ['No Causation', 'X->Y', 'Y->X']
	for i, pattern in enumerate(patterns):
		# Extract data for this pattern only
		pattern_mask = (true_relations == i)
		if not any(pattern_mask):
			continue
			
		# Get predictions for this pattern
		pattern_true = true_relations[pattern_mask]
		pattern_pred = predicted_relations[pattern_mask]
		
		# Calculate correct prediction rate for this specific pattern
		correct_predictions = (pattern_true == pattern_pred)
		
		# Store results
		results[f'For {pattern} data'] = {
			'Sample count': sum(pattern_mask),
			'Predicted as No Causation': sum(pattern_pred == 0) / len(pattern_pred),
			'Predicted as X->Y': sum(pattern_pred == 1) / len(pattern_pred),
			'Predicted as Y->X': sum(pattern_pred == 2) / len(pattern_pred),
			'Correct prediction rate': sum(pattern_true == pattern_pred) / len(pattern_pred)
		}
		
	results['Overall Performance'] = {
		'Total samples': len(true_relations),
		'Accuracy': sum(true_relations == predicted_relations) / len(true_relations)
	}
	
	# Add confusion matrix
	results['Confusion Matrix'] = conf_matrix
	
	return results

def experiment_using_artififial_data(n_samples=1000, n_datasets_per_type=1000):
	# Generate datasets (100 for each type)
	datasets = generate_dataset(n_samples=n_samples, n_datasets_per_type=n_datasets_per_type)
	all_results = dict()
	
	# Simulate results for causal discovery methods (replace this with actual methods)
	true_relations = [d[1] for d in datasets]
	

	# methods = ["DRCD", "MIC", "LiM", "CRACK", "HCM", "GSF"]
	methods = ["DRCD"]

	for method_i, method in enumerate(methods):
		print(f"\nEvaluating method {method_i+1}/{len(methods)}: {method}")
		predicted_relations = []

		for dataset in tqdm(datasets, desc=f"Processing {method}", ncols=100):

			if method == "DRCD":
				drcd = DRCD.DRCD(dataset[0])
				predicted_relations.append(drcd.infer())
			
			# elif method == "MIC":
			# 	predicted_relations.append(MIC().detect_causality(dataset[0][:,0], dataset[0][:,1]))

			# elif method == "LiM":
			# 	model = LiM()
			# 	with redirect_stdout(open(os.devnull, 'w')):
			# 		model.fit(dataset[0], np.array([[1,0]]))
			# 	if not model._adjacency_matrix[0][1] == 0:
			# 		lim_result = 1
			# 	elif not model._adjacency_matrix[1][0] == 0:
			# 		lim_result = 2
			# 	else:
			# 		lim_result = 0
			# 	predicted_relations.append(lim_result)
			# elif method == "CRACK":
			# 	predicted_relations.append(crack_cpp.run_crack(dataset[0][:,0], dataset[0][:,1], os.getcwd()+"/other_methods/crack/code", threshold=0.01))
			# elif method == "HCM":
			# 	predicted_relations.append(HCM.execute(dataset[0]))
			# elif method == "GSF":
			# 	predicted_relations.append(GSF.infer_causal_direction(dataset[0]))


		# Evaluate performance
		results = evaluate_method(true_relations, predicted_relations)
		all_results[method] = results
		
		# Display results
		print("\nEvaluation Results:")
		for section, metrics in results.items():
			print(f"\n{section}:")
			if section == 'Confusion Matrix':
				print("True (rows) vs Predicted (columns):")
				print(metrics)
			else:
				for metric_name, value in metrics.items():
					if isinstance(value, float):
						print(f"{metric_name}: {value:.3f}")
					else:
						print(f"{metric_name}: {value}")

	# Create directory to save results
	results_dir = 'results/artificial'
	os.makedirs(results_dir, exist_ok=True)
	
	# Save results as pickle file
	results_file = os.path.join(results_dir, 'all_results_DRCDonly.pkl')
	with open(results_file, 'wb') as f:
		pickle.dump(all_results, f)
	
	print(f"\nResults saved to {results_file}")

	# textablemake.generate_latex_table(results_file=results_file)
	textablemake.generate_latex_table(results_file=results_file, DRCDonly=True)

	print("Complete!")



if __name__ == '__main__':
	experiment_using_artififial_data()

