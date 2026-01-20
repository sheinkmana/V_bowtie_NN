import numpy as np 
import math 
from scipy.stats import norm
import jax.numpy as jnp

def RMSE(y_true, y_pred):
    '''Compute Root Mean Square Error.'''
    return jnp.sqrt(jnp.mean((y_true - y_pred.reshape(y_true.shape))**2))

def NL(y_true, m, std):
    '''Compute Negative Log-Likelihood.'''
    nll = - jnp.mean(norm.logpdf(y_true.reshape(-1), loc = m.reshape(-1), scale =std.reshape(-1)))
    return nll

# checking if y given from the model is very "different" from truth 
def empirical_coverage(y_true, y_model, std, n_st_devs=2):
    '''Compute empirical coverage within n standard deviations.'''
    i = 0
    for m in range(len(y_true)):
        if y_true[m] > y_model[m] - n_st_devs*std[m] and y_true[m] < y_model[m] + n_st_devs*std[m]:
            i+=1
    return i/len(y_true)


def empirical_coverage_quantile(y_true, y_pred_mean, y_pred_std, quantile=0.95):
    '''Compute empirical coverage for given quantile.'''
    # Compute the z-score for the given quantile
    z = norm.ppf(0.5 + quantile / 2)
    lower = y_pred_mean - z * y_pred_std
    upper = y_pred_mean + z * y_pred_std
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    return coverage
def is_non_decreasing(list):
    return (np.diff(list) >= 0).all() 


# some of the functions needed for the VBNN class

# inverse of the mean of the inverse gamma distribution used for tau/psi
def inv_mean_IG(nu,delta):
    '''Compute the inverse of the mean of the inverse gamma distribution.'''
    assert nu < 0, "Nu should be < 0"
    return math.exp(math.log(-2*nu) - 2*math.log(delta))

# inverse of the mean of the inverse gamma distribution used for Sigma/eta
def inv_mean_IG_eta(alpha,beta):
    '''Compute the inverse of the mean of the inverse gamma distribution for eta.'''
    return math.exp(math.log(alpha) - math.log(beta))


#stable better_sigmoid
def better_sigmoid(x):
    '''Compute a numerically stable sigmoid function.'''
    if -x > np.log(np.finfo(type(x)).max):
        return 0.0    
    a = np.exp(-x)
    return 1.0/ (1.0 + a)

# mean of the PG rv
def pg_mean(x):
    '''Compute the mean of the Polya-Gamma random variable.'''
    try:
        answer = math.exp(math.log(math.exp(x)-1) - math.log(math.exp(x)+1) - math.log(2) - math.log(x))
    except OverflowError:
        answer = 0.5/x
    return answer

# more stable computation of the log of the hyperbolic cosine
def logcosh(x):
    '''Compute a numerically stable log(cosh(x)).'''
    try:
        answer = math.log(math.cosh(x))
    except OverflowError:
        answer = x - math.log(2)
    return answer 

# used in elbo 
def smart_log(x):
    '''Compute x * log(x) with special handling for x = 0.'''
    if x == 0: 
        return 0
    else: 
        return x*math.log(x)
    

# needed when want to reparametrize the parameters of IG
def reparametrize_nu(nu):
    '''Reparametrize nu for inverse gamma distribution.'''
    alpha = -nu
    return alpha

def reparametrize_delta(delta):
    '''Reparametrize delta for inverse gamma distribution.'''
    beta = delta**2/2
    return beta



