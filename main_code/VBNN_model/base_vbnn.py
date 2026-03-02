import numpy as np
import math
from typing import Optional
from numpy.typing import NDArray

from scipy.stats import norm, invgamma, bernoulli
from sklearn.linear_model import LinearRegression, RidgeCV
from .utils import (inv_mean_IG, inv_mean_IG_eta, better_sigmoid, 
                   pg_mean, logcosh, smart_log, reparametrize_nu, reparametrize_delta)
import jax.random as random
from numpyro_models import model_numpyro_multilayer, do_svi


class VBNNCore:
    """
    Base class for Variational Bayesian Neural Networks.
    Handles architecture setup, weight initialization, forward passes, and prediction.
    """
    def __init__(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        D_H: int, 
        L: int, 
        T: float, 
        wb_mode: str, 
        big_S: float,
        big_B: float,
        beta_eta_h_prior: float,
        sample_size: 'Optional[int]' = None
    ):
        
        self.x = x.reshape(x.shape[0], x.shape[1], 1)
        self.y = y
        self.N, self.D_in, _ = self.x.shape
        self.D_out = self.y.shape[1]
        
        self.sample_size = sample_size

        # Architecture dimensions: [Input, Hidden*L, Output]
        self.D = [self.D_in] + [D_H] * L + [self.D_out]
        self.L = L
        self.T = T
        self.mode = wb_mode

        linear_regr = LinearRegression()
        linear_regr.fit(x, y)
        fitted_linr = linear_regr.predict(x)
        residuals_linr = y.reshape(fitted_linr.shape) - fitted_linr
        beta = residuals_linr.var()
        # delta = np.max(np.abs(linear_regr.coef_))**0.5
        delta = min(np.max(np.abs(linear_regr.coef_))**0.5, 10)
        self.s_0 = np.abs(np.asarray(linear_regr.intercept_).item())**0.5
        self.delta_tau_prior = np.copy(delta)/(self.L**0.5) 
        # self.delta_psi_prior = np.array([delta*self.D[1]**0.5/(self.D[0]**0.5)] + [delta]*self.L)
        self.delta_psi_prior = np.array([1/(self.D[k]**0.5) for k in range(self.L+2)])

        if wb_mode == 'numpyro_init':
            model_kwargs = {
                'L': L,
                'D_H': D_H,
                'D_out': 1,
                'prw': delta,
                'prb': self.s_0,
                'prs':  min(beta/4, 30)
            }

            maxiter =  500
            rng_key, _ = random.split(random.PRNGKey(np.random.randint(0, 10000)))
            self.inf_params, _ = do_svi(model_numpyro_multilayer, rng_key, maxiter,  x, y.reshape(-1), model_kwargs)



        
        # Hyperparameters
        if self.sample_size:
            self.S = [np.array([big_S*np.eye(self.D[i+1])]*self.sample_size) for i in range(self.L)]
        else:
            self.S = [np.array([big_S*np.eye(self.D[i+1])]*self.N) for i in range(self.L)]

        self.big_b = [np.array([big_B * np.eye(self.D[i] + 1)] * self.D[i+1]) for i in range(self.L + 1)]
        self.beta_eta_h_prior = beta_eta_h_prior
        self.beta_eta_o_prior = min(beta/4, 30)  # Add this line or set to a separate argument if needed
        
        # Fixed Priors
        self.alpha_eta_h_prior = 2.0
        self.alpha_eta_o_prior = 2.0
        self.nu_tau_prior = -1.5
        self.nu_psi_prior = -1.5
   
        
        # Global Variance Params
        self.alpha_h = self.alpha_eta_h_prior
        self.beta_h = np.full((self.L, self.D[1]), self.beta_eta_h_prior, dtype=np.float64)
        self.alpha_0 = self.alpha_eta_o_prior
        self.beta_0 = np.full(self.D[self.L+1],0, dtype=np.float64) # Initialized later
        
        # Hierarchical Priors
        self.nu_glob = np.full(self.L +1, self.nu_tau_prior).astype(np.float64)
        self.delta_glob = np.sqrt(2*(-self.nu_glob -1)*invgamma.rvs(a=reparametrize_nu(self.nu_tau_prior), loc=0, scale=reparametrize_delta(self.delta_tau_prior), size = self.L +1))
        
        self.nu_loc =[np.full((self.D[i+1], self.D[i]), self.nu_psi_prior).astype(np.float64) for i in range(self.L + 1)]
        self.delta_loc = [np.sqrt(2*(-self.nu_loc[i]-1)*invgamma.rvs(a=reparametrize_nu(self.nu_psi_prior), loc=0, scale=reparametrize_delta(self.delta_psi_prior[i]), size =(self.D[i+1], self.D[i]))) for i in range(self.L + 1)]
        

        if self.sample_size:
                        # Caching
                # Initialize cached_means: list of arrays with shape (N, D[i], 1)
            self.cached_means: list[NDArray] = [
                np.zeros((self.sample_size, self.D[i], 1)) for i in range(self.L + 1)
            ]
            
            # Initialize cached_aats: list of arrays with shape (N, D[i], D[i])
            self.cached_aats: list[NDArray] = [
                np.zeros((self.sample_size, self.D[i], self.D[i])) for i in range(self.L + 1)
            ]

        else:
            # Caching
                # Initialize cached_means: list of arrays with shape (N, D[i], 1)
            self.cached_means: list[NDArray] = [
                np.zeros((self.N, self.D[i], 1)) for i in range(self.L + 1)
            ]
            
            # Initialize cached_aats: list of arrays with shape (N, D[i], D[i])
            self.cached_aats: list[NDArray] = [
                np.zeros((self.N, self.D[i], self.D[i])) for i in range(self.L + 1)
            ]

        self.cache_valid = False
           # Prediction caching variables
          # Initialize cached_means: list of arrays with shape (N, D[i], 1)

        self.b_predict, self.M_predict, self.S_predict, self.rho_predict,self.A,  self.A_predict = None, None, None, None, None, None

        self.cached_means_predict: list[NDArray] = [
            np.zeros((self.N, self.D[i], 1)) for i in range(self.L + 1)
        ]
        
        # Initialize cached_aats: list of arrays with shape (N, D[i], D[i])
        self.cached_aats_predict: list[NDArray] = [
            np.zeros((self.N, self.D[i], self.D[i])) for i in range(self.L + 1)
        ]

        self.cache_valid_predict = False
        # Initialize
        self._init_priors_and_weights()
        self.epoch_no = None
        self.elbo_total, self.elbo_pred = None, None

    def _init_priors_and_weights(self): 
        self.m_h = []      # Hidden layer means
        self.rho = []           
        self.M = []
        self.b = []
        
        if self.sample_size:
              #uniformly sample S indices
            self.sample_indices =  np.sort(np.random.choice(self.N, self.sample_size, replace = False))
            self.x_sample = np.copy(self.x[self.sample_indices])
            self.y_sample = np.copy(self.y[self.sample_indices])
            x_set = np.copy(self.x_sample)
            indices_range = self.sample_size
        else:
            x_set = np.copy(self.x)
            indices_range = self.N

        if self.mode == 'standard_init':
            for k in range(self.L):
                m_h_k = np.zeros((self.D[k+1], 1, self.D[k]+1))  # Correct shape
                for i in range(self.D[k+1]):
                    m_h_k[i, 0, 0] = 0                           # Bias
                    m_h_k[i, 0, 1:] = norm.rvs(loc=0, scale=np.sqrt(2/(self.D[k] + self.D[k+1])), size=self.D[k])             # Weights
                self.m_h.append(m_h_k)
                self.rho.append(np.array([[better_sigmoid((self.m_h[-1][i][0, 0] + self.m_h[-1][i][:, 1:]@x_set[n]).item()/self.T) for i in range(self.D[k+1])] for n in range(indices_range)]))
                self.M.append(np.array([self.m_h[-1].squeeze()[:, 1:]*self.rho[-1][j].reshape(self.D[k+1], 1) for j in range(indices_range)]))
                self.b.append(np.array([(self.m_h[-1].squeeze()[:,0]*self.rho[-1][j]).reshape(self.D[k+1],1) for j in range(indices_range)]))
                x_set = np.array([np.copy(self.M[k][n])@np.copy(x_set[n]) + np.copy(self.b[k][n]) for n in range(indices_range)])

            
             # Output layer
            
            x_stand = np.copy(x_set.squeeze())
            temp_m_o = [np.array([]) for _ in range(self.D[self.L+1])]
            beta_0 = []
            for i in range(self.D[self.L+1]):
                if self.sample_size:
                    reg = RidgeCV().fit(x_stand, self.y_sample[:, i])
                    temp_m_o[i] = np.hstack([reg.intercept_, reg.coef_]).reshape(1, self.D[self.L] + 1) 
                    fitted_linr = reg.predict(x_stand)  
                    residuals_linr = self.y_sample[:, i].reshape(fitted_linr.shape) - fitted_linr
                else:    
                    reg = RidgeCV().fit(x_stand, self.y[:, i])
                    temp_m_o[i] = np.hstack([reg.intercept_, reg.coef_]).reshape(1, self.D[self.L] + 1) 
                    fitted_linr = reg.predict(x_stand)  
                    residuals_linr = self.y[:, i].reshape(fitted_linr.shape) - fitted_linr
                beta_0.append(residuals_linr.var())
            self.m_o = np.array(temp_m_o)
            self.beta_0 =   np.full(self.D[self.L+1],min(beta_0, self.beta_eta_o_prior), dtype=np.float64)

        if self.mode == 'standard_init_spike':
            for k in range(self.L):
                # Create shape (D[k+1], 1, D[k]+1)
                m_h_k = np.zeros((self.D[k+1], 1, self.D[k]+1))
                
                # Sparsity probability: more connections for smaller layers
                p = 1 / (1 + np.sqrt(self.D[k]))
                sigma_b = 0.9
                for i in range(self.D[k+1]):
                    # Bias = 0
                    m_h_k[i, 0, 0] = norm.rvs(loc=0, scale=sigma_b, size=1)
                    # m_h_k[i, 0, 0] = 0
                    
                    # Spike-and-slab weights:
                    # With prob (1-p): weight = 0
                    # With prob p:     weight ~ N(0, 1/(p·D[k]))
                    spike_mask = bernoulli.rvs(p=p, size=self.D[k])
                    weights = norm.rvs(
                        loc=0, 
                        scale=np.sqrt((1-sigma_b**2) / (p * self.D[k])),  # Larger variance for active weights
                        size=self.D[k]
                    ) * spike_mask  # Zero out inactive weights
                    
                    m_h_k[i, 0, 1:] = weights
                
                self.m_h.append(m_h_k)
                
                # Compute rho, M, b as in standard_init
                self.rho.append(np.array([[
                    better_sigmoid((self.m_h[-1][i][0, 0] + self.m_h[-1][i][:, 1:]@x_set[n]).item()/self.T) 
                    for i in range(self.D[k+1])
                ] for n in range(indices_range)]))
                
                self.M.append(np.array([
                    self.m_h[-1].squeeze()[:, 1:]*self.rho[-1][j].reshape(self.D[k+1], 1) 
                    for j in range(indices_range)
                ]))
                
                self.b.append(np.array([
                    (self.m_h[-1].squeeze()[:,0]*self.rho[-1][j]).reshape(self.D[k+1],1) 
                    for j in range(indices_range)
                ]))
                
                x_set = np.array([
                    np.copy(self.M[k][n])@np.copy(x_set[n]) + np.copy(self.b[k][n]) 
                    for n in range(indices_range)
                ])
            
            # Output layer: use RidgeCV (same as standard_init)
            x_stand = np.copy(x_set.squeeze())
            temp_m_o = [np.array([]) for _ in range(self.D[self.L+1])]
            beta_0 = []
            for i in range(self.D[self.L+1]):
                if self.sample_size:
                    reg = RidgeCV().fit(x_stand, self.y_sample[:, i])
                    temp_m_o[i] = np.hstack([reg.intercept_, reg.coef_]).reshape(1, self.D[self.L] + 1)
                    fitted_linr = reg.predict(x_stand)
                    residuals_linr = self.y_sample[:, i].reshape(fitted_linr.shape) - fitted_linr
                else:
                    reg = RidgeCV().fit(x_stand, self.y[:, i])
                    temp_m_o[i] = np.hstack([reg.intercept_, reg.coef_]).reshape(1, self.D[self.L] + 1)
                    fitted_linr = reg.predict(x_stand)
                    residuals_linr = self.y[:, i].reshape(fitted_linr.shape) - fitted_linr
                beta_0.append(residuals_linr.var())
            
            self.m_o = np.array(temp_m_o)
            self.beta_0 = np.full(self.D[self.L+1], min(beta_0 + [self.beta_eta_o_prior]), dtype=np.float64)

        if self.mode == 'standard_init_laplace':
            for k in range(self.L):
                # Create shape (D[k+1], 1, D[k]+1)
                m_h_k = np.zeros((self.D[k+1], 1, self.D[k]+1))
                sigma_b = 0.9
                
                for i in range(self.D[k+1]):
                    # Bias = 0
                    m_h_k[i, 0, 0] = norm.rvs(loc=0, scale=sigma_b, size=1)
                    
                    # Laplace weights: sharp peak at zero, heavy tails
                    # scale = sqrt(1 / (2 * D[k]))
                    weights = np.random.laplace(
                        loc=0, 
                        scale=np.sqrt((1-sigma_b**2)/ (2 * self.D[k])),
                        size=self.D[k]
                    )
                    
                    m_h_k[i, 0, 1:] = weights
                
                self.m_h.append(m_h_k)
                
                # Compute rho, M, b as in standard_init
                self.rho.append(np.array([[
                    better_sigmoid((self.m_h[-1][i][0, 0] + self.m_h[-1][i][:, 1:]@x_set[n]).item()/self.T) 
                    for i in range(self.D[k+1])
                ] for n in range(indices_range)]))
                
                self.M.append(np.array([
                    self.m_h[-1].squeeze()[:, 1:]*self.rho[-1][j].reshape(self.D[k+1], 1) 
                    for j in range(indices_range)
                ]))
                
                self.b.append(np.array([
                    (self.m_h[-1].squeeze()[:,0]*self.rho[-1][j]).reshape(self.D[k+1],1) 
                    for j in range(indices_range)
                ]))
                
                x_set = np.array([
                    np.copy(self.M[k][n])@np.copy(x_set[n]) + np.copy(self.b[k][n]) 
                    for n in range(indices_range)
                ])
            
            # Output layer: use RidgeCV (same as standard_init)
            x_stand = np.copy(x_set.squeeze())
            temp_m_o = [np.array([]) for _ in range(self.D[self.L+1])]
            beta_0 = []
            for i in range(self.D[self.L+1]):
                if self.sample_size:
                    reg = RidgeCV().fit(x_stand, self.y_sample[:, i])
                    temp_m_o[i] = np.hstack([reg.intercept_, reg.coef_]).reshape(1, self.D[self.L] + 1)
                    fitted_linr = reg.predict(x_stand)
                    residuals_linr = self.y_sample[:, i].reshape(fitted_linr.shape) - fitted_linr
                else:
                    reg = RidgeCV().fit(x_stand, self.y[:, i])
                    temp_m_o[i] = np.hstack([reg.intercept_, reg.coef_]).reshape(1, self.D[self.L] + 1)
                    fitted_linr = reg.predict(x_stand)
                    residuals_linr = self.y[:, i].reshape(fitted_linr.shape) - fitted_linr
                beta_0.append(residuals_linr.var())
            
            self.m_o = np.array(temp_m_o)
            self.beta_0 = np.full(self.D[self.L+1], min(beta_0 + [self.beta_eta_o_prior]), dtype=np.float64)

        if self.mode == 'numpyro_init':
            for k in range(self.L):
                self.m_h.append(np.array(np.hstack((self.inf_params[f'nn_b{k+1}_auto_loc'].reshape(-1, 1), (self.inf_params[f'nn_w{k+1}_auto_loc'].T))).reshape(self.D[k+1], 1, self.D[k]+1)))
                self.rho.append(np.array([[better_sigmoid((self.m_h[-1][i][0, 0] + self.m_h[-1][i][:, 1:]@x_set[n]).item()/self.T) for i in range(self.D[k+1])] for n in range(indices_range)]))
                self.M.append(np.array([self.m_h[-1].squeeze()[:, 1:]*self.rho[-1][j].reshape(self.D[k+1], 1) for j in range(indices_range)]))
                self.b.append(np.array([(self.m_h[-1].squeeze()[:,0]*self.rho[-1][j]).reshape(self.D[k+1],1) for j in range(indices_range)]))
                x_set = np.array([np.copy(self.M[k][n])@np.copy(x_set[n]) + np.copy(self.b[k][n]) for n in range(indices_range)])
            self.m_o = np.hstack((self.inf_params[f'nn_b{self.L+1}_auto_loc'].reshape(-1, 1), (self.inf_params[f'nn_w{self.L+1}_auto_loc'].T))).reshape(self.D[self.L+1], 1, self.D[self.L]+1)
            self.beta_0 =  np.full(self.D[self.L+1], self.beta_eta_o_prior, dtype=np.float64)

        if self.mode == 'sparse_init':
            m_slot_h =[np.array([]) for _ in range(self.D[1])]             
            for i in range(self.D[1]):
                vector_p = bernoulli.rvs(p = 1/(1 + np.sqrt(self.D[0])), size = (1, self.D[0]))
                part_m_1 = norm.rvs(loc = 0, scale = np.sqrt(2/np.sqrt(self.D[0])), size = (1, self.D[0]))*vector_p
                delta = (x_set.squeeze().max(axis=0) - x_set.squeeze().min(axis=0))*0.05                
                use = np.random.uniform(low = x_set.squeeze().min(axis=0) - delta, high = x_set.squeeze().max(axis=0) + delta).reshape(self.D[0], 1)
                m_slot_h[i] = np.hstack((-part_m_1@use, part_m_1)) # type: ignore
                
            self.m_h.append(np.array(m_slot_h))
            self.rho.append(np.array([[better_sigmoid((self.m_h[-1][i][0, 0] + self.m_h[-1][i][:, 1:]@x_set[n]).item()/self.T) for i in range(self.D[1])] for n in range(indices_range)]))
            self.M.append([self.m_h[-1].squeeze()[:, 1:]*self.rho[-1][j].reshape(self.D[1], 1) for j in range(indices_range)])
            self.b.append([(self.m_h[-1].squeeze()[:,0]*self.rho[-1][j]).reshape(self.D[1],1) for j in range(indices_range)])

            x_set = np.array([np.copy(self.M[0][n])@np.copy(x_set[n]) + np.copy(self.b[0][n]) for n in range(indices_range)])
#  just set them to map the previous nodes to a node of the next layer, e.g. m^w_{l,d} is a vector of zeros with a single one entry and for m^b_{l,d}, s is just the min. Or if you want some generalization of this that allows more sparsity as you move up.
            for k in range(1, self.L):
                m_slot_h = np.zeros((self.D[k+1], 1, self.D[k]+1))
                m_slot_h[:, 0, 0] = x_set.squeeze().min(axis=0)
                for i in range(self.D[k+1]):
                    m_slot_h[i, 0, i+1] = 1.0

                self.m_h.append(np.array(m_slot_h))
                self.rho.append(np.array([[better_sigmoid((self.m_h[-1][i][0, 0] + self.m_h[-1][i][:, 1:]@x_set[n]).item()/self.T) for i in range(self.D[k+1])] for n in range(indices_range)]))
                self.M.append([self.m_h[-1].squeeze()[:, 1:]*self.rho[-1][j].reshape(self.D[k+1], 1) for j in range(indices_range)])
                self.b.append([(self.m_h[-1].squeeze()[:,0]*self.rho[-1][j]).reshape(self.D[k+1],1) for j in range(indices_range)])
                x_set = np.array([np.copy(self.M[k][n])@np.copy(x_set[n]) + np.copy(self.b[k][n]) for n in range(indices_range)])
                

            x_stand = np.copy(x_set.squeeze())
            temp_m_o = [np.array([]) for _ in range(self.D[self.L+1])]
            beta_0 = []
            for i in range(self.D[self.L+1]):
                if self.sample_size:
                    reg = RidgeCV().fit(x_stand, self.y_sample[:, i])
                    temp_m_o[i] = np.hstack([reg.intercept_, reg.coef_]).reshape(1, self.D[self.L] + 1) 
                    fitted_linr = reg.predict(x_stand)  
                    residuals_linr = self.y_sample[:, i].reshape(fitted_linr.shape) - fitted_linr
                else:    
                    reg = RidgeCV().fit(x_stand, self.y[:, i])
                    temp_m_o[i] = np.hstack([reg.intercept_, reg.coef_]).reshape(1, self.D[self.L] + 1) 
                    fitted_linr = reg.predict(x_stand)  
                    residuals_linr = self.y[:, i].reshape(fitted_linr.shape) - fitted_linr
                beta_0.append(residuals_linr.var())
            self.m_o = np.array(temp_m_o)
            self.beta_0 =   np.full(self.D[self.L+1],min(beta_0, self.beta_eta_o_prior), dtype=np.float64)

        if self.mode == 'laplace' or self.mode == 'spikeslab':
            for k in range(self.L):
                m_slot_h =[np.array([]) for _ in range(self.D[k+1])]             
                # m_slot_h =[[] for _ in range(self.D[k+1])]             
                for i in range(self.D[k+1]):
                    if self.mode == 'laplace':
                        part_m_1 = np.random.laplace(loc = 0, scale = np.sqrt(2/self.D[k]), size =  (1, self.D[k]))
                    # else:
                    elif self.mode == 'spikeslab':
                        vector_p = bernoulli.rvs(p = 1/(1 + np.sqrt(self.D[k])), size = (1, self.D[k]))
                        part_m_1 = norm.rvs(loc = 0, scale = np.sqrt(2/np.sqrt(self.D[k])), size = (1, self.D[k]))*vector_p
                    delta = (x_set.squeeze().max(axis=0) - x_set.squeeze().min(axis=0))*0.05                
                    use = np.random.uniform(low = x_set.squeeze().min(axis=0) - delta, high = x_set.squeeze().max(axis=0) + delta).reshape(self.D[k], 1)
                    m_slot_h[i] = np.hstack((-part_m_1@use, part_m_1)) # type: ignore
                    
                self.m_h.append(np.array(m_slot_h))
                print(f'shape of m_h for layer {k}: {self.m_h[-1].shape}')
                print(f'Layer sizes: {self.D[k]} -> {self.D[k+1]}')
                self.rho.append(np.array([[better_sigmoid((self.m_h[-1][i][0, 0] + self.m_h[-1][i][:, 1:]@x_set[n]).item()/self.T) for i in range(self.D[k+1])] for n in range(indices_range)]))
                self.M.append([self.m_h[-1].squeeze()[:, 1:]*self.rho[-1][j].reshape(self.D[k+1], 1) for j in range(indices_range)])
                self.b.append([(self.m_h[-1].squeeze()[:,0]*self.rho[-1][j]).reshape(self.D[k+1],1) for j in range(indices_range)])
                x_set = np.array([np.copy(self.M[k][n])@np.copy(x_set[n]) + np.copy(self.b[k][n]) for n in range(indices_range)])
                

            x_stand = np.copy(x_set.squeeze())
            temp_m_o = [np.array([]) for _ in range(self.D[self.L+1])]
            beta_0 = []
            for i in range(self.D[self.L+1]):
                if self.sample_size:
                    reg = RidgeCV().fit(x_stand, self.y_sample[:, i])
                    temp_m_o[i] = np.hstack([reg.intercept_, reg.coef_]).reshape(1, self.D[self.L] + 1) 
                    fitted_linr = reg.predict(x_stand)  
                    residuals_linr = self.y_sample[:, i].reshape(fitted_linr.shape) - fitted_linr
                else:    
                    reg = RidgeCV().fit(x_stand, self.y[:, i])
                    temp_m_o[i] = np.hstack([reg.intercept_, reg.coef_]).reshape(1, self.D[self.L] + 1) 
                    fitted_linr = reg.predict(x_stand)  
                    residuals_linr = self.y[:, i].reshape(fitted_linr.shape) - fitted_linr
                beta_0.append(residuals_linr.var())
            self.m_o = np.array(temp_m_o)
            self.beta_0 =   np.full(self.D[self.L+1],min(beta_0, self.beta_eta_o_prior), dtype=np.float64)

    def _init_sample_params(self):
        if self.sample_size is None:
            return
        
        self.M, self.b, self.rho= [], [], []
        #uniformly sample S indices
        self.sample_indices =  np.sort(np.random.choice(self.N, self.sample_size, replace = False))
        self.x_sample = np.copy(self.x[self.sample_indices])
        self.y_sample = np.copy(self.y[self.sample_indices])
        x_set = np.copy(self.x_sample)
        for k in range(self.L):
            self.rho.append(np.array([[better_sigmoid((self.m_h[k][i][0, 0] + np.array(self.m_h[k])[i][:, 1:]@x_set[n]).item()/self.T) for i in range(self.D[k+1])] for n in range(self.sample_size)]))
            self.M.append([np.array(self.m_h[k]).squeeze()[:, 1:]*self.rho[-1][j].reshape(self.D[k+1], 1) for j in range(self.sample_size)])
            self.b.append([(np.array(self.m_h[k]).squeeze()[:,0]*self.rho[-1][j]).reshape(self.D[k+1],1) for j in range(self.sample_size)])
            x_set = np.array([np.copy(self.M[k][n])@np.copy(x_set[n]) + np.copy(self.b[k][n]) for n in range(self.sample_size)])

    
    def _compute_forward_pass(self):
        if self.cache_valid:
            return
        
        if self.sample_size:
            self.cached_means[0] = self.x_sample  # Shape: (N, D[0], 1)
            self.cached_aats[0] = np.array([x @ x.T for x in self.x_sample])
              # Shape: (N, D[0], D[0])
            indices_range = self.sample_size
        else:
            self.cached_means[0] = self.x  # Shape: (N, D[0], 1)
            self.cached_aats[0] = np.array([x @ x.T for x in self.x])  # Shape: (N, D[0], D[0])
            indices_range = self.N
        # Forward pass through hidden layers
        for layer in range(1, self.L + 1):
            means = np.zeros((indices_range, self.D[layer], 1))
            aats = np.zeros((indices_range, self.D[layer], self.D[layer]))
            
            for n in range(indices_range):
                prev_mean = self.cached_means[layer-1][n]
                prev_aat = self.cached_aats[layer-1][n] 
                
                # Mean computation
                means[n] = self.b[layer-1][n] + self.M[layer-1][n] @ prev_mean
                
                # Second moment computation
                aats[n] = (self.S[layer-1][n] + 
                          self.b[layer-1][n] @ self.b[layer-1][n].T +
                          self.M[layer-1][n] @ prev_aat @ self.M[layer-1][n].T +
                          self.M[layer-1][n] @ prev_mean @ self.b[layer-1][n].T + 
                          self.b[layer-1][n] @ prev_mean.T @ self.M[layer-1][n].T)
            
            self.cached_means[layer] = means
            self.cached_aats[layer] = aats
            
        self.cache_valid = True
    

    def _elbo_tau(self):
        elbo_tau = 0.5*sum([inv_mean_IG(self.nu_glob[k], self.delta_glob[k])*(self.delta_glob[k]**2- self.delta_tau_prior**2) for k in range(self.L+1)]) \
            +2*sum([self.nu_glob[k]*math.log(self.delta_glob[k]) for k in range(self.L+1)]) - 2*(self.L+1)*self.nu_tau_prior*math.log(self.delta_tau_prior)
        return elbo_tau
    
 
    def _elbo_psi(self):
        elbo_psi= 0.5*sum([inv_mean_IG(self.nu_loc[k][i][j], self.delta_loc[k][i][j])*(self.delta_loc[k][i][j]**2 - self.delta_psi_prior[k]**2) for k in range(self.L+1) for i in range(self.D[k+1]) for j in range(self.D[k])])\
            + 2*sum([self.nu_loc[k][i][j]*math.log(self.delta_loc[k][i][j]) for k in range(self.L+1) for i in range(self.D[k+1]) for j in range(self.D[k])])
            # + 2*sum([self.nu_loc[k][i][j]*math.log(self.delta_loc[k][i][j]) for k in range(self.L+1) for i in range(self.D[k+1]) for j in range(self.D[k])]) - 2*sum([self.D[k+1]*self.D[k]*self.nu_psi_prior*math.log(self.delta_psi_prior[k]) for k in range(self.L+1)])
        return elbo_psi
  
     
    def _elbo_eta(self):
        elbo_eta = sum([(self.beta_h[k][i]- self.beta_eta_h_prior)*inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for k in range(self.L) for i in range(self.D[k+1]) ])\
            - sum([self.alpha_h*math.log(self.beta_h[k][i]) for k in range(self.L) for i in range(self.D[k+1])])\
                + sum([(self.beta_0[i]-self.beta_eta_o_prior)*(inv_mean_IG_eta(self.alpha_0, self.beta_0[i])) for i in range(self.D[self.L+1])])\
                    - sum([self.alpha_0*math.log(self.beta_0[i]) for i in range(self.D[self.L+1])])
        return elbo_eta
   
    def _elbo_WB(self):
        elbo_Wb = 0.5*sum([np.linalg.slogdet(self.big_b[k][i])[1] for k in range(self.L+1) for i in range(self.D[k+1])])\
            - 0.5*sum([(self.m_h[k][i][0,0]**2 + self.big_b[k][i][0,0])/self.s_0**2 for k in range(self.L) for i in range(self.D[k+1])])\
                - 0.5*sum([(self.m_o[i][0,0]**2 + self.big_b[self.L][i][0,0])/self.s_0**2 for i in range(self.D[self.L+1])])\
                    - 0.5*sum([(self.m_h[k][i][0,1+j]**2+ self.big_b[k][i][1+j,1+j])*inv_mean_IG(self.nu_glob[k], self.delta_glob[k])*inv_mean_IG(self.nu_loc[k][i][j], self.delta_loc[k][i][j]) for k in range(self.L) for i in range(self.D[k+1])  for j in range(self.D[k])])\
                        - 0.5*sum([(self.m_o[i][0,j+1]**2 + self.big_b[self.L][i][j+1,j+1])*inv_mean_IG(self.nu_glob[self.L], self.delta_glob[self.L])*inv_mean_IG(self.nu_loc[self.L][i][j], self.delta_loc[self.L][i][j]) for i in range(self.D[self.L+1]) for j in range(self.D[self.L])])
        return elbo_Wb
    
    def _elbo_agamma(self):
        if self.A is None:
            raise ValueError("A_predict must be initialized before calling prediction (via initializing VBNN_algorithm).")

        temp_sum = 0
        temp_sum_out = 0
        if self.sample_size is not None:
            indices_range = self.sample_size
            y = np.copy(self.y_sample)
        else:
            indices_range = self.N
            y = np.copy(self.y)
        for j in range(indices_range):
            for i in range(self.D[self.L+1]):
                a_mean = self.cached_means[self.L][j]
                aat_mean = self.cached_aats[self.L][j]
                temp_sum_out += -0.5*inv_mean_IG_eta(self.alpha_0, self.beta_0[i])*\
                    ((y[j][i] - self.m_o[i][0,0] -self.m_o[i][:,1:]@a_mean)**2\
                    + (self.big_b[self.L][i][1:,1:]*aat_mean).sum()\
                    + 2*self.big_b[self.L][i][0, 1:]@a_mean+ self.big_b[self.L][i][0,0]\
                    + ((self.m_o[i][:,1:].T@self.m_o[i][:,1:])*(aat_mean - a_mean@a_mean.T)).sum())

            for k in range(self.L):
                a_mean_prev =  self.cached_means[k][j]
                aat_prev = self.cached_aats[k][j]
                aat_current = self.cached_aats[k+1][j]
                a_current = self.cached_means[k+1][j]
             
                for i in range(self.D[k+1]):
                    aat_shifted = self.b[k][j][i]*a_mean_prev + (aat_prev@self.M[k][j][i]).reshape(self.D[k],1)
                    big_trace = self.big_b[k][i][0,0] + self.m_h[k][i][0,0]**2 + 2*(self.big_b[k][i][0,1:] +self.m_h[k][i][0,0]*self.m_h[k][i][:,1:])@a_mean_prev\
                        + ((self.big_b[k][i][1:,1:] + self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*(aat_prev).T).sum()
                    small_trace = self.m_h[k][i][0,0]**2 + 2*self.m_h[k][i][0,0]*self.m_h[k][i][:,1:]@a_mean_prev + ((self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*(a_mean_prev@a_mean_prev.T).T).sum()
                    temp_sum += -0.5*inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*((self.rho[k][j][i]*(self.m_h[k][i][:,1:]@a_mean_prev +self.m_h[k][i][0,0])- a_current[i])**2\
                                                                                       + aat_current[i,i] - a_current[i]**2 + self.rho[k][j][i]*big_trace -self.rho[k][j][i]**2*small_trace\
                                                                                        + self.rho[k][j][i]*self.m_h[k][i][:,1:]@(a_current[i]*a_mean_prev - aat_shifted))\
                                    + 1/self.T*(self.rho[k][j][i] - 0.5)*(self.m_h[k][i][:,1:]@a_mean_prev +self.m_h[k][i][0,0]) - 1/(2*self.T**2)*pg_mean(self.A[k][j][i])*big_trace\
                                        - smart_log(self.rho[k][j][i]) - smart_log((1-self.rho[k][j][i])) - logcosh(self.A[k][j][i]*0.5) + 0.5*pg_mean(self.A[k][j][i])*self.A[k][j][i]**2

                temp_sum += 0.5*np.linalg.slogdet(self.S[k][j])[1]    
       
        
        elbo_agamma = temp_sum + temp_sum_out
        
        return elbo_agamma*float(self.N/indices_range)
    
    def elbo(self):
        if self.elbo_total is None:
            raise ValueError("elbo_total must be initialized before calling elbo (via initializing VBNN_algorithm).")
        # self.elbo_total.append((self.elbo_tau() + self.elbo_psi() + self.elbo_eta() + self.elbo_WB() + self.elbo_agamma()))
        self.elbo_total.append((self._elbo_tau() + self._elbo_psi() + self._elbo_eta() + self._elbo_WB() + self._elbo_agamma()).item()) # type: ignore

   
    def _new_delta(self,):
        self.delta_tau_prior = np.sqrt((self.L+1)*self.nu_tau_prior/np.sum([self.nu_glob[k]/self.delta_glob[k]**2 for k in range(self.L+1)]))
 

