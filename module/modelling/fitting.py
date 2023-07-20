from scipy.optimize import curve_fit
from scipy.stats import chisquare
from .models import power_law, linear
import numpy as np

def fit(function, x, y, yerr, x_range=None, **kwargs):
    """
    General fitting function using scipy.optimize.curve_fit
    """
    popt, pcov = curve_fit(function, x, y,
                            p0=None, # Initial guess
                            sigma=yerr, # Uncertainty in y
                            absolute_sigma=False, # Uncertainty is relative
                            check_finite=True, # Check for NaNs
                            bounds=(-np.inf, np.inf), # Bounds on parameters
                            method=None, # Method for optimization
                            # nan_policy='omit', # How to handle NaNs
                            **kwargs) # Additional arguments passed to leastsq/least_squares
    
    model_values = None
    if x_range is not None:
        model_values = generate_model_values(function, popt, x_range)
    return popt, pcov, model_values

def fit_linear(x, y, yerr):
    """
    Fit a linear function to the data
    """
    x_range = (x.min(), x.max())
    popt, pcov, model_values = fit(linear, x, y, yerr, x_range)
    slope, intercept = popt
    return slope, intercept, pcov, model_values

def fit_power_law(x, y, yerr):
    """
    Linearly fit the logged data, equivalent to fitting a power law
        but produces a better fit as we minimise the least
        squares of the log of the data rather than the data itself.
    """
    # Convert to log space and fit
    x_range = (np.log10(x.min()), np.log10(x.max()))
    popt, pcov, model_values = fit(linear, np.log10(x), np.log10(y), 1/np.log10(yerr), x_range)
    coefficient, exponent = popt

    # Convert back to linear space
    model_values = (10**model_values[0], 10**model_values[1])
    coefficient = 10**coefficient

    # Print the best fit
    print(f'fitted power law: y = {coefficient:.2f}*x^{exponent:.2f}')
    return coefficient, exponent, pcov, model_values

def fit_power_law_linear(x, y, yerr):
    """
    Fit a power law to the data itself.
    However, fit_power_law might be more appropriate.
    """
    x_range = (x.min(), x.max())
    popt, pcov, model_values = fit(power_law, x, y, yerr, x_range)
    coefficient, exponent = popt
    return coefficient, exponent, pcov, model_values

def chi(x, y, function, popt):
    """
    Calculate the chi^2 value for the fit
    """
    return chisquare(y, function(x, *popt), ddof=2)

def r_squared(x, y, function, popt):
    """
    Calculate the r^2 value for the fit
    """
    residuals = y - function(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot)

def generate_model_values(function, params, x_range, n_points=100):
    """
    Generate a set of values from a model
    """
    x = np.linspace(*x_range, n_points)
    y = function(x, *params)
    return x, y