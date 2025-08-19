import numpy as np
import cvxpy as cp

class PortfolioOptimization:
    def __init__(self, historical_returns, predicted_returns):
        """
        Initialize the PortfolioOptimization class.

        Parameters:
        - historical_returns: DataFrame of historical asset returns (used for covariance and risk metrics)
        - predicted_returns: Array or Series of predicted asset returns (from ML model)
        """
        self.returns = historical_returns  # Historical returns for risk metrics
        self.predicted_returns = predicted_returns.T  # Predicted returns for optimization (ensure correct shape)
        self.cov_matrix = historical_returns.cov()  # Covariance matrix for portfolio volatility


    def optimize_portfolio(self, risk_profile=None, target_return=None):
        """
        Optimize portfolio weights to minimize volatility, subject to constraints.

        Parameters:
        - risk_profile: String indicating user's risk profile (optional)
        - target_return: Minimum target return for the portfolio (optional)

        Returns:
        - List of optimized weights (percentages) for each asset, or None if infeasible
        """
        num_assets = len(self.predicted_returns)

        # Define optimization variable for asset weights
        weights = cp.Variable(num_assets)  

        # Objective: Minimize portfolio volatility (variance)
        portfolio_volatility = cp.quad_form(weights, self.cov_matrix)
        objective = cp.Minimize(portfolio_volatility)

        # Constraints: weights sum to 1, no short selling (weights >= 0)
        constraints = [
            cp.sum(weights) == 1,
            weights >= 0
        ]

        # If no explicit target return, map risk profile to target return
        if target_return is None and risk_profile is not None:
            target_return = self.risk_profile_to_target_return(risk_profile)
        
        # Add target return constraint if specified
        if target_return is not None:
            constraints.append(cp.matmul(weights, self.predicted_returns) >= target_return)

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # If solution is feasible, normalize and return weights as percentages
        if problem.status not in ["infeasible", "unbounded"]:
            optimized_weights = weights.value
            normalized_weights = optimized_weights / np.sum(optimized_weights)
            min_vol_return = float(np.dot(normalized_weights, self.predicted_returns))
            print("Target Return:", min_vol_return)
            return (normalized_weights * 100).tolist() 
        else:
            print("Optimization failed:", problem.status)
            return None


    def calculate_sharpe_ratio(self, weights, risk_free_rate):
        """
        Calculate the Sharpe ratio for the portfolio.

        Parameters:
        - weights: Portfolio weights (array-like)
        - risk_free_rate: Risk-free rate (should match return frequency, e.g., weekly)

        Returns:
        - Sharpe ratio (float)
        """
        portfolio_return = np.dot(self.predicted_returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        sharpe_period = (portfolio_return - risk_free_rate) / portfolio_volatility
        return sharpe_period * np.sqrt(26)  # Annualize Sharpe ratio if returns are biweekly


    def calculate_sortino_ratio(self, weights, risk_free_rate):
        """
        Calculate the Sortino ratio for the portfolio.

        Parameters:
        - weights: Portfolio weights (array-like)
        - risk_free_rate: Risk-free rate (should match return frequency)

        Returns:
        - Sortino ratio (float or np.nan if no downside deviation)
        """
        portfolio_return = np.dot(self.predicted_returns, weights)
        portfolio_returns = self.returns @ weights  # Historical portfolio returns
        downside_returns = portfolio_returns[portfolio_returns < 0]  # Returns below zero
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0
        sortino_period = (portfolio_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else np.nan
        return sortino_period * np.sqrt(26)
    

    @staticmethod
    def risk_profile_to_target_return(risk_profile):
        """
        Map a risk profile string to a target return value.

        Parameters:
        - risk_profile: String describing the user's risk profile

        Returns:
        - Target return (float) or None if not matched
        """
        risk_perf = None
        profiles = [
            "Very Aggressive", "Very Conservative", "Conservative", "Moderate", "Aggressive"
        ]
        for profile in profiles:
            if profile.lower() in risk_profile.lower():
                risk_perf = profile
                break
        # Mapping risk profile to target return (example values, adjust as needed)
        risk_profile_mapping = {
            'Very Aggressive': 0.01,
            'Very Conservative': 0.002,
            'Conservative': 0.004,
            'Moderate': 0.006,
            'Aggressive': 0.008
        }

        return risk_profile_mapping.get(risk_perf)
