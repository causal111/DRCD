# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as opt
from scipy.special import expit  # Sigmoid function
from sklearn.linear_model import LinearRegression, LogisticRegression

class CausalityDetector:
    def __init__(self, lambda_pen=0.1):
        """
        Initialize the MixedCausalityDetector with a penalty parameter.
        
        Parameters:
        -----------
        lambda_pen : float, optional
            L0 penalty parameter for MIC score (default: 0.1)
        """
        self.lambda_pen = lambda_pen
    
    def _negative_log_likelihood_continuous(self, X, beta=None, Y=None, b=None):
        """
        Compute negative log likelihood for continuous variable (Laplace distribution).
        
        Parameters:
        -----------
        X : array-like
            Continuous variable data
        beta : float, optional
            Coefficient for parent variable (if any)
        Y : array-like, optional
            Parent variable data (if any)
        b : float, optional
            Scale parameter for Laplace distribution
            
        Returns:
        --------
        float
            Negative log likelihood value
        """
        if b is None:
            b = self._estimate_scale_parameter(X)
        
        if beta is not None and Y is not None:
            residuals = X - beta * Y
        else:
            residuals = X
            
        return np.sum(np.abs(residuals) / b) + len(X) * np.log(2 * b)
    
    def _negative_log_likelihood_binary(self, Y, beta=None, X=None):
        """
        Compute negative log likelihood for binary variable (Logistic distribution).
        
        Parameters:
        -----------
        Y : array-like
            Binary variable data
        beta : float, optional
            Coefficient for parent variable (if any)
        X : array-like, optional
            Parent variable data (if any)
            
        Returns:
        --------
        float
            Negative log likelihood value
        """
        if beta is not None and X is not None:
            p = expit(beta * X)
        else:
            p = np.mean(Y) * np.ones_like(Y)
            
        return -np.sum(Y * np.log(p + 1e-10) + (1 - Y) * np.log(1 - p + 1e-10))
    
    def _estimate_scale_parameter(self, X):
        """
        Estimate the scale parameter b for Laplace distribution.
        
        Parameters:
        -----------
        X : array-like
            Continuous variable data
            
        Returns:
        --------
        float
            Estimated scale parameter
        """
        return np.mean(np.abs(X - np.median(X)))
    
    def _compute_optimal_parent_set(self, Xi, X_minus_i=None):
        """
        Compute the optimal potential parent set for variable Xi.
        This is a simplified version since we only have two variables.
        
        Parameters:
        -----------
        Xi : array-like
            The target variable
        X_minus_i : array-like or None
            The other variable (potential parent)
            
        Returns:
        --------
        tuple
            (optimal_beta, minimal_likelihood)
        """
        if X_minus_i is None or len(X_minus_i) == 0:
            # No potential parent
            if np.all((Xi == 0) | (Xi == 1)):  # Binary variable
                return 0, self._negative_log_likelihood_binary(Xi)
            else:  # Continuous variable
                return 0, self._negative_log_likelihood_continuous(Xi)
        
        # With potential parent
        if np.all((Xi == 0) | (Xi == 1)):  # Binary variable
            # Find optimal beta for binary target
            res = opt.minimize_scalar(
                lambda beta: self._negative_log_likelihood_binary(Xi, beta, X_minus_i)
            )
            return res.x, res.fun
        else:  # Continuous variable
            # Find optimal beta for continuous target
            res = opt.minimize_scalar(
                lambda beta: self._negative_log_likelihood_continuous(Xi, beta, X_minus_i)
            )
            return res.x, res.fun
    
    def _compute_wi(self, Xi, X_minus_i=None):
        """
        Compute the scale parameter wi for MIC score as described in the paper.
        
        Parameters:
        -----------
        Xi : array-like
            The target variable
        X_minus_i : array-like or None
            The potential parent variable
            
        Returns:
        --------
        float
            Scale parameter wi
        """
        _, wi = self._compute_optimal_parent_set(Xi, X_minus_i)
        return wi
    
    def _compute_MIC(self, Xi, Pa_Xi, wi):
        """
        Compute the Mixed Information Criterion (MIC) score.
        
        Parameters:
        -----------
        Xi : array-like
            The target variable
        Pa_Xi : array-like or None
            The parent set of Xi
        wi : float
            Scale parameter for normalization
            
        Returns:
        --------
        float
            MIC score
        """
        if Pa_Xi is None or len(Pa_Xi) == 0:
            # No parent
            if np.all((Xi == 0) | (Xi == 1)):  # Binary variable
                LL = self._negative_log_likelihood_binary(Xi)
            else:  # Continuous variable
                LL = self._negative_log_likelihood_continuous(Xi)
            pen = 0  # No parameters, no penalty
        else:
            # With parent
            if np.all((Xi == 0) | (Xi == 1)):  # Binary variable
                beta, _ = self._compute_optimal_parent_set(Xi, Pa_Xi)
                LL = self._negative_log_likelihood_binary(Xi, beta, Pa_Xi)
            else:  # Continuous variable
                beta, _ = self._compute_optimal_parent_set(Xi, Pa_Xi)
                LL = self._negative_log_likelihood_continuous(Xi, beta, Pa_Xi)
            pen = self.lambda_pen  # One parameter (beta), apply penalty
        
        return (1/wi) * LL + pen
    
    def detect_causality(self, X, Y):
        """
        Detect causality between X (continuous) and Y (binary) using MIC score.
        
        Parameters:
        -----------
        X : array-like
            Continuous variable data
        Y : array-like
            Binary variable data
            
        Returns:
        --------
        int
            0 for Independent, 1 for X->Y, 2 for Y->X
        dict
            Detailed scores for each model
        """
        # Ensure X is continuous and Y is binary
        if np.all((X == 0) | (X == 1)) and not np.all((Y == 0) | (Y == 1)):
            X, Y = Y, X
            swap = True
        else:
            swap = False
        
        # Compute scale parameters
        w_X = self._compute_wi(X, Y)
        w_Y = self._compute_wi(Y, X)
        
        # Model 1: Independent
        score_X_indep = self._compute_MIC(X, None, w_X)
        score_Y_indep = self._compute_MIC(Y, None, w_Y)
        score_indep = score_X_indep + score_Y_indep
        
        # Model 2: X -> Y
        score_X_XY = self._compute_MIC(X, None, w_X)  # X has no parent
        score_Y_XY = self._compute_MIC(Y, X, w_Y)     # Y has X as parent
        score_XY = score_X_XY + score_Y_XY
        
        # Model 3: Y -> X
        score_X_YX = self._compute_MIC(X, Y, w_X)     # X has Y as parent
        score_Y_YX = self._compute_MIC(Y, None, w_Y)  # Y has no parent
        score_YX = score_X_YX + score_Y_YX
        
        scores = {
            0: score_indep,  # Independent
            1: score_XY,     # X -> Y
            2: score_YX      # Y -> X
        }
        
        best_model = min(scores.items(), key=lambda x: x[1])[0]
        
        if swap and best_model > 0:
            # Swap back the causal direction
            best_model = 3 - best_model  # 1 becomes 2, 2 becomes 1
        
        return best_model
    
    def fit(self, X, Y):
        """
        Fit the causal model to the data and return the causal direction.
        
        Parameters:
        -----------
        X : array-like
            First variable data
        Y : array-like
            Second variable data
            
        Returns:
        --------
        str
            Causal relationship description
        """
        X = np.asarray(X).flatten()
        Y = np.asarray(Y).flatten()
        
        result, scores = self.detect_causality(X, Y)
        
        if result == 0:
            return "X and Y are independent", scores
        elif result == 1:
            return "X causes Y", scores
        else:  # result == 2
            return "Y causes X", scores

# Example usage
if __name__ == "__main__":
    # Generate example data: X -> Y
    np.random.seed(42)
    X = np.random.normal(0, 1, 1000)
    prob_Y = expit(1.5 * X)
    Y = np.random.binomial(1, prob_Y)
    
    detector = MixedCausalityDetector()
    result, scores = detector.detect_causality(X, Y)
    
    print(f"Detected causal relationship: {result}")
    print(f"Model scores: {scores}")
    print(detector.fit(X, Y))