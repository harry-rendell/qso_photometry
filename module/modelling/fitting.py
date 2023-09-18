from scipy.optimize import curve_fit, minimize
from scipy.stats import chisquare
from . import models
import numpy as np

def fit(function, x, y, yerr, n_model_points=None, least_sq_kwargs={}, **curve_fit_kwargs):
    """
    General fitting function using scipy.optimize.curve_fit
    """
    p0             = curve_fit_kwargs['p0'] if 'p0' in curve_fit_kwargs else None
    absolute_sigma = curve_fit_kwargs['absolute_sigma'] if 'absolute_sigma' in curve_fit_kwargs else False
    check_finite   = curve_fit_kwargs['check_finite'] if 'check_finite' in curve_fit_kwargs else None
    method         = curve_fit_kwargs['method'] if 'method' in curve_fit_kwargs else 'trf'

    x_fit_bounds   = curve_fit_kwargs['x_fit_bounds'] if 'x_fit_bounds' in curve_fit_kwargs else (-np.inf, np.inf)
    value_range = curve_fit_kwargs['value_range'] if 'value_range' in curve_fit_kwargs else (x.min(), x.max())
    mask = (x_fit_bounds[0] <= x) & (x <= x_fit_bounds[1])

    if 'bounds' in curve_fit_kwargs:
        # recast
        bounds = ([b[0] for b in curve_fit_kwargs['bounds']], [b[1] for b in curve_fit_kwargs['bounds']])
    else:
        bounds = (-np.inf, np.inf)

    popt, pcov = curve_fit(function, x[mask], y[mask],
                            p0=p0, # Initial guess
                            sigma=yerr[mask], # Uncertainty in y
                            absolute_sigma=absolute_sigma, # Uncertainty is relative
                            check_finite=check_finite, # Check for NaNs
                            method=method, # Method for optimization
                            bounds=bounds,
                            # nan_policy='omit', # How to handle NaNs
                            **least_sq_kwargs) # Additional arguments passed to leastsq/least_squares
    
    # Generate values from the model
    model_values = generate_model_values(function, popt, x, n_model_points=n_model_points, value_range=value_range)
    rms_error = rmse(y[mask], generate_model_values(function, popt, x[mask])[0])
    print(f'RMSE: {rms_error:.5f}')
    return popt, pcov, model_values

def fit_minimize(fit_function, cost_function, x, y, yerr, n_model_points=None, **kwargs):
    """
    Given a fit function and a cost function, find the best fit using scipy.optimize.minimize
    """
    x0 = kwargs['x0'] if 'x0' in kwargs else None # Initial guess
    method = kwargs['method'] if 'method' in kwargs else 'Nelder-Mead' # Method for optimization
    bounds = kwargs['bounds'] if 'bounds' in kwargs else None # Bounds for optimization
    tol = kwargs['tol'] if 'tol' in kwargs else 1e-6 # Tolerance for optimization

    x_fit_bounds   = kwargs['x_fit_bounds'] if 'x_fit_bounds' in kwargs else (-np.inf, np.inf)
    value_range = kwargs['value_range'] if 'value_range' in kwargs else (x.min(), x.max())

    mask = (x_fit_bounds[0] <= x) & (x <= x_fit_bounds[1])

    # Wrap the cost function so that it takes the correct arguments
    cf = lambda params, *args: cost_function(fit_function, params, *args)
    res = minimize(cf, x0, args=(x[mask], y[mask], yerr[mask]), method=method, bounds=bounds, tol=tol)

    popt = res['x']
    model_values = generate_model_values(fit_function, popt, x, n_model_points=n_model_points, value_range=value_range)
    return popt, None, model_values

def fit_linear(x, y, yerr, **kwargs):
    """
    Fit a linear function to the data
    """
    popt, pcov, model_values = fit(models.linear, x, y, yerr, **kwargs)
    slope, intercept = popt
    return slope, intercept, pcov, model_values

def fit_power_law(x, y, yerr, **kwargs):
    """
    Linearly fit the logged data, equivalent to fitting a power law
        but produces a better fit as we minimise the least
        squares of the log of the data rather than the data itself.
    """
    # Convert to log space and fit
    popt, pcov, model_values = fit(models.linear, np.log10(x), np.log10(y), 1/np.log10(yerr), **kwargs)
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
    popt, pcov, model_values = fit(models.power_law, x, y, yerr)
    coefficient, exponent = popt
    return coefficient, exponent, pcov, model_values

def fit_broken_power_law(x, y, yerr, least_sq_kwargs={}, **kwargs):
    """
    Fit a broken power law to the data
    """
    #x, A, x_b, a_1, a_2, delta=1
    popt, pcov, model_values = fit(models.bkn_pow_smooth, x, y, yerr, least_sq_kwargs=least_sq_kwargs, **kwargs)
    amplitude, break_point, index_1, index_2 = popt
    return amplitude, break_point, index_1, index_2, pcov, model_values

def fit_DRW_SF(x, y, yerr, **kwargs):
    """
    Fit a damped random walk structure function to the data
    """
    popt, pcov, model_values = fit(models.DRW_SF, x, y, yerr, **kwargs)
    tau, SF_inf = popt
    return tau, SF_inf, pcov, model_values

def fit_mod_DRW_SF(x, y, yerr, **kwargs):
    """
    Fit a damped random walk structure function to the data
    """
    popt, pcov, model_values = fit(models.mod_DRW_SF, x, y, yerr, **kwargs)
    tau, SF_inf, beta = popt
    return tau, SF_inf, beta, pcov, model_values

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

def rmse(y, y_pred):
    """
    Calculate the root mean squared error
    """
    return np.sqrt(np.mean((y - y_pred)**2))

def generate_model_values(function, params, x, n_model_points=None, value_range=None):
    """
    Generate a set of values from a model.
    If we pass in a number of points, generate model values
        over the range of x values with that number of points.
    Otherwise, generate model values over the range of x data values
    """
    if n_model_points is None:
        y = function(x, *params)
    
    else:
        x = np.linspace(value_range[0], value_range[1], int(n_model_points))
        y = function(x, *params)
    
    return x, y

def cost_function(f, params, *args):
    """
    Cost function of f
    """
    x, y, yerr = args
    y_pred =  f(x, *params)
    if y_pred.min() < 0:
        return np.inf
    return np.sum((np.log10(y) - np.log10(y_pred))**2)
    # return np.sum((y - y_pred)**2) # In theory this should arrive at the same ish result as curve_fit