# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd

HCM_DIR = "./other_methods/AAAI2022-HCM-main"
sys.path.append(HCM_DIR)

from mixed_causal import mixed_causal
from const import DEFAULT_MODEL_PARA, DEFAULT_LGBM_PARA

base_model_para = {**DEFAULT_MODEL_PARA, **DEFAULT_LGBM_PARA}
base_model_para["base_model"] = "lgbm"


def execute(X):
	df = pd.DataFrame(X, columns=["continuous_var", "categorical_var"])

	variable_info = {
		'continuous': [0],  # Column 0 is continuous variable
		'categorical': [1]  # Column 1 is categorical variable
	}


	# Encode categorical variables (convert to numeric type)
	df["categorical_var"] = df["categorical_var"].astype(int)

	# Modify `X_encode` (store each column as 2D numpy array in a list)
	X_encode = [df.iloc[:, i].to_numpy().reshape(-1, 1) for i in range(df.shape[1])]


	# Execute HCM
	try:
		result = mixed_causal(df, X_encode, DEFAULT_MODEL_PARA, base_model_para)
		if result[2][1][0] == 1:
			r = 2
		else:
			r = 1
	except ValueError:
		r = 0

	return r