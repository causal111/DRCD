# -*- coding: utf-8 -*-

import numpy as np
import collections
import pandas as pd
from scipy import stats
from densratio import densratio
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')


class DRCD():
	"""
	Density Ratio-based Causal Discovery (DRCD) is a method for discovering causal relationships 
	between a continuous variable and a discrete variable.
	
	This class implements algorithms to determine causal relationships based on density ratio
	estimation and monotonicity testing.
	
	Parameters
	----------
	X : array-like
		Input data with two columns. The first column 'x' contains continuous values,
		and the second column 'y' contains discrete values.
	
	Attributes
	----------
	_X : pandas.DataFrame
		DataFrame containing the input data with columns 'x' and 'y'.
	_dv : list
		The two most common discrete values in the 'y' column.
	_threshold : float
		Significance threshold for statistical tests.
	"""

	def __init__(self, X):
		self._X = pd.DataFrame(X, columns=["x","y"])

	@property
	def X(self):
		return self._X

	@X.setter
	def X(self, value):
		self._X = value

	@property
	def dv(self):
		return self._dv

	@property
	def threshold(self):
		return self._threshold

	@threshold.setter
	def threshold(self, value):
		self._threshold = value


	def determine_the_discrete_values(self):
		mc = collections.Counter(self._X["y"]).most_common()
		if len(mc) < 2:
			raise ValueError("The data must contain at least two distinct discrete values")
		self._dv = [mc[0][0], mc[1][0]]
		return self._dv


	def determine_ks_2samp_pval(self):
		self.determine_the_discrete_values()
		first_x = self._X[self._X["y"]==self._dv[0]]["x"]
		second_x = self._X[self._X["y"]==self._dv[1]]["x"]
		self._ks_2samp_pval = stats.ks_2samp(first_x, second_x)[1]
		return self._ks_2samp_pval


	def infer(self, threshold = 0.05, n_points = 1000, graph = False):
		"""
		Infer causal relationship between variables.
		
		Parameters
		----------
		threshold : float, optional
			Significance threshold for statistical tests, default is 0.05.
		n_points : int, optional
			Number of points to use in density ratio estimation, default is 1000.
		graph : bool, optional
			Whether to display a graph of the density ratio, default is False.
			
		Returns
		-------
		int
			Causality type:
			0 = No causality detected
			1 = x causes y 
			2 = y causes x
			
		Notes
		-----
		The method first checks for statistical differences using the two-sample Kolmogorov-Smirnov test.
		If significant differences are not found, we conclude that there is no causal relationship.
		If significant differences are found, it estimates the density ratio and tests for monotonicity of the density ratio.
		If the density ratio is monotonic, we conclude that X causes Y (X->Y); otherwise, we conclude that Y causes X (Y->X).
		"""

		self._threshold = threshold
		self._n_points = n_points
		self._graph = graph

		self.determine_the_discrete_values()

		self._presence_of_causality = self.has_causality()

		if not self._presence_of_causality:
			self._causality = 0
			return self._causality

		density_ratio_result = self.estimate_density_ratio()
		if density_ratio_result is None:
			# Handle the case where density ratio estimation failed
			self._causality = 0
			return self._causality

		monotonicity = self.is_monotonic()
		if monotonicity:
			self._causality = 1
		else:
			self._causality = 2

		return self._causality


	def has_causality(self):
		self.determine_ks_2samp_pval()
		return self._ks_2samp_pval < self._threshold


	def estimate_density_ratio(self):
		"""
		Estimate density ratio between the two most common discrete values.
		
		Returns
		-------
		tuple or None
			Tuple containing x sequence and density ratio values, or None if estimation fails.
		"""
		first_x = self._X[self._X["y"]==self._dv[0]]["x"].to_numpy()
		second_x = self._X[self._X["y"]==self._dv[1]]["x"].to_numpy()

		x_min = max(np.min(first_x), np.min(second_x))
		x_max = min(np.max(first_x), np.max(second_x))

		if x_min > x_max:
			print("Warning: No overlapping range for density ratio estimation.")
			return None

		x_seq = np.linspace(x_min, x_max, self._n_points)

		try:
			# Set fixed random seed to ensure the densratio algorithm produces consistent results.
			# This prevents getting different outcomes from identical input data since the algorithm uses randomization internally.
			np.random.seed(1)
			densratio_func = densratio(first_x, second_x, verbose=False)
			y_seq = densratio_func.compute_density_ratio(x_seq)

			self._x_seq = x_seq
			self._y_seq = y_seq
			self._d_seq = np.diff(self._y_seq)

			if self._graph:
				plt.plot(x_seq, y_seq, "o")
				plt.xlabel("x")
				plt.ylabel("Density Ratio")
				plt.show()

			return self._x_seq, self._y_seq
		except Exception as e:
			print(f"Density ratio estimation failed: {str(e)}")
			return None


	def is_monotonic(self):
		"""
		Test whether the density ratio exhibits a monotonic trend.

		This method evaluates whether the sequence of differences in the estimated density ratio
		(stored in self._d_seq) shows a statistically significant deviation from zero mean.
		If the population mean of the differences is significantly different from zero,
		it suggests a consistent upward or downward trend in the density ratio, i.e., monotonicity.

		Specifically, the test checks:
		- Null hypothesis H0: The mean of the difference sequence is zero.
		- If the p-value is smaller than the threshold (self._threshold), we reject H0 and conclude
			that the density ratio likely exhibits a monotonic pattern (either increasing or decreasing).

		Returns
		-------
		bool
			True if a monotonic pattern is detected (mean of differences significantly differs from zero),
			False otherwise.
		"""

		return stats.ttest_1samp(self._d_seq, popmean=0)[1] < self._threshold


