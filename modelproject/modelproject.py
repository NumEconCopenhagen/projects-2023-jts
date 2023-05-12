import numpy as np
from scipy import optimize
import random

def n_ss_solow(variables, s_K, s_H, n, g, delta, alpha, varphi):
    """
    Args:
    variables (list or tuple): Contains two variables to be solved for: physical capital and human capital
    s_K               (float): Savings rate in physical capital
    s_H               (float): Savings rate in human capital
    n                 (float): Population growth rate
    g                 (float): TFP growth rate
    delta             (float): Depreciation rate
    alpha             (float): Physical capital share in the production function
    varphi            (float): Human capital share in the production function
    
    Returns:
    Physical capital and human capital equations in steady state
    """
    # Variables to be solved for: physical capital and human capital
    k, h = variables
    
    # Check for edge cases
    if k <= 0 or h <= 0:
        # Return a very large residual to indicate a poor solution
        return [np.inf, np.inf]

    # Sets the Solow equations in steady state
    n_ss_solow_k = (1 / ((1 + n) * (1 + g))) * (s_K * k**alpha * h**varphi - (n + g + delta + n * g) * k)
    n_ss_solow_h = (1 / ((1 + n) * (1 + g))) * (s_H * k**alpha * h**varphi - (n + g + delta + n * g) * h)

    # Return equations
    return n_ss_solow_k, n_ss_solow_h


def multi_start(num_guesses=100, bounds=[1e-5, 10], fun=n_ss_solow, args= None, method='hybr'):
    """
    Performs multi-start optimization to find the steady state solutions for k and h.
    
    Args:
    num_guesses     (int): The number of random initial guesses, default = 100
    bounds        (tuple): The bounds for the random initial guesses, default = [1e-5, 10]
    fun        (function): The function to be optimized, default = n_ss_solow
    args          (tuple): The tuple of arguments for the function, default = None
    method       (method): The optimization method to use, default = 'hybr'
    
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
        sol = optimize.root(fun = fun, x0 = initial_guess, args = args, method = method)
        
        # Calculate the residual norm (Euclidean norm) of the current solution
        residual_norm = np.linalg.norm(sol.fun)

        # If the residual norm is smaller than the current smallest residual, update the steady state of k and h and the smallest residual
        if residual_norm < smallest_residual:
            smallest_residual = residual_norm
            ms_ss_k, ms_ss_h = sol.x
    
    # Return optimal solutions 
    return ms_ss_k, ms_ss_h, smallest_residual

def null_clines(s_K, s_H, g, n, alpha, varphi, delta, Max = 10, N = 500):
    """
    Args:
    s_K               (float): Savings rate in physical capital
    s_H               (float): Savings rate in human capital
    n                 (float): Population growth rate
    g                 (float): TFP growth rate
    delta             (float): Depreciation rate
    alpha             (float): Physical capital share in the production function
    varphi            (float): Human capital share in the production function
    Max               (float): Maximum value of k
    N                   (int): Number of values of k
    
    Returns:
    Null-clines for physical capital and human capital
    """

    # Create a vector for N values of k from 0 to Max 
    k_vec = np.linspace(1e-5, Max, N)

    # Create two empty N-size arrays to store the null-clines
    h_vec_k, h_vec_h = np.empty(N), np.empty(N)

    # Set root_error to False
    root_error = False

    # Iterate through each value of k in k_vec
    for i, k in enumerate(k_vec):
        
        # Determine the null-clines 
        null_k = lambda h: - n_ss_solow((k, h), s_K, s_H, n, g, delta, alpha, varphi)[0]
        null_h = lambda h: - n_ss_solow((k, h), s_K, s_H, n, g, delta, alpha, varphi)[1]

        try:
            # Find roots for the null-clines
            sol_k = optimize.root_scalar(f = null_k, method = 'brentq', bracket = [1e-20, 20])
            sol_h = optimize.root_scalar(f = null_h, method = 'brentq', bracket = [1e-20, 20])

            # Save the roots
            h_vec_k[i], h_vec_h[i] = sol_k.root, sol_h.root
    
        except ValueError:
            if root_error == False:
                print('Due to f(a)f(b)>0, the method failed to find roots for some or all values of k')
                root_error = True  # Set the flag to True
            h_vec_k[i], h_vec_h[i] = np.nan, np.nan

    # Return array of solutions
    return k_vec, h_vec_k, h_vec_h

def find_intersection(x, y, z):
    # Find index of intersection where the sign of (y - z) changes
    idx = np.where(np.diff(np.sign(y - z)))

    # Return value of x, y and z at index
    return x[idx], y[idx], z[idx]


