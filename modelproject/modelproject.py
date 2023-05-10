import numpy as np
from scipy import optimize
import random

def n_ss(variables, s_K, s_H, n, g, delta, alpha, varphi):
    """
    Args:
    variables (list or tuple): Contains two variables to be solved for: capital and human capital
    s_K               (float): Savings rate in capital
    s_H               (float): Savings rate in human capital
    n                 (float): Population growth rate
    g                 (float): TFP growth rate
    delta             (float): Depreciation rate
    alpha             (float): Capital share in the production function
    varphi            (float): Human capital share in the production function
    
    Returns:
    Capital and human capital in steady state
    """
    # Variables to be solved for: capital and human capital
    k, h = variables
    
    # Check for edge cases
    if k <= 0 or h <= 0:
        return [1e10, 1e10]  # Return a very large residual to indicate a poor solution

    # Sets the Solow equations in steady state
    n_ss_solow_k = (1 / ((1 + n) * (1 + g))) * (s_K * k**alpha * h**varphi - (n + g + delta + n * g) * k)
    n_ss_solow_h = (1 / ((1 + n) * (1 + g))) * (s_H * k**alpha * h**varphi - (n + g + delta + n * g) * h)

    return n_ss_solow_k, n_ss_solow_h


def multi_start(num_guesses=100, bounds=[0.1, 10], fun=n_ss, args= None, method='hybr'):
    """
    Performs multi-start optimization to find the steady state solutions for k and h.
    
    Args:
    num_guesses     (int): The number of random initial guesses, default=100
    bounds        (tuple): The bounds for the random initial guesses, default=[0.1, 10]
    fun        (function): The function to be optimized, default=n_ss
    args          (tuple): The tuple of arguments for the function, default= None
    method       (method): The optimization method to use, default='hybr'
    
    Returns:
    Prints the steady state values for k and h, and the residual of the function
    """
    # Initialize the smallest residual as infinity
    smallest_residual = np.inf

    # Generate a list of random numbers within the specified bounds
    random_samples = list(np.random.uniform(low=bounds[0], high=bounds[1], size=num_guesses))

    # Loop through each random initial guess
    for i in range(num_guesses):
        # Select a random pair of numbers from the list of random samples
        initial_guess = random.sample(random_samples, 2)

        # Solve the optimization problem with the current initial guess
        sol = optimize.root(fun=fun, x0=initial_guess, args=args, method=method)
        
        # Calculate the residual norm (Euclidean norm) of the current solution
        residual_norm = np.linalg.norm(sol.fun)

        # If the residual norm is smaller than the current smallest residual, update the steady state of k and h and the smallest residual
        if residual_norm < smallest_residual:
            smallest_residual = residual_norm
            ms_ss_k, ms_ss_h = sol.x

    return ms_ss_k, ms_ss_h, smallest_residual