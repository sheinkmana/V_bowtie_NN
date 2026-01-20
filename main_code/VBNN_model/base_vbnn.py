import numpy as np
import math
from typing import Optional
from numpy.typing import NDArray

from scipy.stats import norm, invgamma, bernoulli
from sklearn.linear_model import LinearRegression, RidgeCV
from utils import (inv_mean_IG, inv_mean_IG_eta, better_sigmoid, 
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
        self.delta_tau_prior = np.copy(delta)
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
    
    # def _compute_forward_pass_predict(self, x_new):
    #     if self.cache_valid_predict:
    #         return

    #     N_pred = x_new.shape[0]

    #     # Ensure b_predict and M_predict are initialized
    #     if self.b_predict is None or self.M_predict is None or self.S_predict is None:
    #         raise ValueError("b_predict and M_predict must be initialized before calling _compute_forward_pass_predict.")

    #     # Layer 0 (input)
    #     self.cached_means_predict[0] = x_new  # Shape: (N_pred, D[0], 1)
    #     self.cached_aats_predict[0] = np.array([x @ x.T for x in x_new])  # Shape: (N_pred, D[0], D[0])

    #     # Forward pass through hidden layers
    #     for layer in range(1, self.L + 1):
    #         means = np.zeros((N_pred, self.D[layer], 1))
    #         aats = np.zeros((N_pred, self.D[layer], self.D[layer]))

    #         for n in range(N_pred):
    #             prev_mean = self.cached_means_predict[layer-1][n]
    #             prev_aat = self.cached_aats_predict[layer-1][n]

    #             # Mean computation
    #             means[n] = self.b_predict[layer-1][n] + self.M_predict[layer-1][n] @ prev_mean 

    #             # Second moment computation  
    #             aats[n] = (self.S_predict[layer-1][n] + 
    #                     self.b_predict[layer-1][n] @ self.b_predict[layer-1][n].T +
    #                     self.M_predict[layer-1][n] @ prev_aat @ self.M_predict[layer-1][n].T +
    #                     self.M_predict[layer-1][n] @ prev_mean @ self.b_predict[layer-1][n].T + 
    #                     self.b_predict[layer-1][n] @ prev_mean.T @ self.M_predict[layer-1][n].T)

    #         self.cached_means_predict[layer] = means
    #         self.cached_aats_predict[layer] = aats

    #     self.cache_valid_predict = True


    # def predict(self, x_for_pred, epochs_pred, rate_pred = 0.00001):
    #     if self.A_predict is None:
    #         raise ValueError("A_predict must be initialized before calling prediction (via initializing VBNN_algorithm).")

    #     pred_time_start = time.time()
    #     N_pred = x_for_pred.shape[0]
    #     x_new = np.ndarray.copy(x_for_pred).reshape(N_pred, self.D[0], 1)
    #     self.elbo_pred = []

    #     self.M_predict = []
    #     self.b_predict = []
    #     self.rho_predict = []
    #     x_init = np.ndarray.copy(x_new)
    #     for k in range(self.L):
    #         self.rho_predict.append(np.array([[better_sigmoid((self.m_h[k][i][0, 0] + self.m_h[k][i][:, 1:]@x_init[n]).item()/self.T) for i in range(self.D[k+1])] for n in range(N_pred)]))
    #         self.M_predict.append([(self.m_h[k].squeeze()[:, 1:]*self.rho_predict[k][j].reshape(self.D[k+1],1))for j in range(N_pred)])
    #         self.b_predict.append([(self.m_h[k].squeeze()[:,0]*self.rho_predict[k][j]).reshape(self.D[k+1], 1) for j in range(N_pred)])
    #         x_init = np.array([np.copy(self.M_predict[k][n])@np.copy(x_init[n]) + np.copy(self.b_predict[k][n]) for n in range(N_pred)]).reshape(N_pred, self.D[k+1], 1)

    #     self.S_predict = [np.array([0.01*np.eye(self.D[k+1])]*N_pred) for k in range(self.L)]
        
    #     self.cache_valid_predict = False
    #     self._compute_forward_pass_predict(x_new)
        
    #     for k in range(self.L):    
    #         self.A_predict[k] = []
    #         for j in range(N_pred):
    #             tobm = self.cached_means_predict[k][j]
    #             # tobm = self.mean_finder_predict(x_new, k, j)
    #             big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.ndarray.copy(self.cached_aats_predict[k][j])))))
    #             self.A_predict[k].append([1/self.T*math.sqrt(((self.big_b[k][i] + self.m_h[k][i].T@self.m_h[k][i])*big_matrix).sum()) for i in range(self.D[k+1])])


    #     for ep_pred in range(epochs_pred):

        
    #         for k in reversed(range(self.L)):
    #             if k == self.L -1:
    #                 b_h_diag_predict =np.diag([1/inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for i in range(self.D[k+1])])
    #                 for j in range(N_pred):
    #                     self.S_predict[k][j]= np.ndarray.copy(b_h_diag_predict)
    #                     rho_dot_w = self.m_h[k].squeeze()[:,1:]*self.rho_predict[k][j].reshape(self.D[k+1],1)
    #                     rho_dot_b = (self.m_h[k].squeeze()[:,0]*self.rho_predict[k][j]).reshape(self.D[k+1],1)
    #                     self.b_predict[k][j] = np.copy(rho_dot_b)
    #                     self.M_predict[k][j] = np.copy(rho_dot_w)
    #             else:
    #                 # Step 1 - parameters for new a
    #                 b_h_diag_predict = np.diag([inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for i in range(self.D[k+1])])
    #                 for j in range(N_pred):
    #                     self.S_predict[k][j]= solve(b_h_diag_predict - self.M_predict[k+1][j].T@solve(self.S_predict[k+1][j], np.eye(self.D[k+1]), assume_a = 'pos')@self.M_predict[k+1][j]\
    #                                         + np.sum([(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho_predict[k+1][j][i] + 1/self.T**2*pg_mean(self.A_predict[k+1][j][i]))*(self.big_b[k+1][i][1:, 1:] + self.m_h[k+1][i][:, 1:].T@self.m_h[k+1][i][:, 1:]) for i in range(self.D[k+2])], axis = 0),  np.eye(self.D[k+1]), assume_a = 'pos')
    #                     rho_dot_w = self.m_h[k].squeeze()[:,1:]*self.rho_predict[k][j].reshape(self.D[k+1],1)
    #                     rho_dot_b = (self.m_h[k].squeeze()[:,0]*self.rho_predict[k][j]).reshape(self.D[k+1],1)
    #                     sum_for_b =  np.sum([-(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho_predict[k+1][j][i]\
    #                                         + 1/self.T**2*pg_mean(self.A_predict[k+1][j][i]))*(self.big_b[k+1][i][1:,:1] + self.m_h[k+1][i][0,0]*self.m_h[k+1][i][:,1:].T) + 1/self.T*(self.rho_predict[k+1][j][i] - 0.5)*self.m_h[k+1][i][:,1:].T for i in range(self.D[k+2])], axis =0)
    #                     self.b_predict[k][j] = self.S_predict[k][j]@(b_h_diag_predict@rho_dot_b + self.M_predict[k+1][j].T@solve(self.S_predict[k+1][j],  np.eye(self.D[k+1]), assume_a = 'pos')@self.b_predict[k+1][j] + sum_for_b)
    #                     self.M_predict[k][j] = self.S_predict[k][j]@b_h_diag_predict@rho_dot_w 
            
    #         self.cache_valid_predict = False
    #         self._compute_forward_pass_predict(x_new)


    #         for k in range(self.L):    
    #             self.A_predict[k] = []
    #             for j in range(N_pred):
    #                 tobm =  self.cached_means_predict[k][j]
    #                 big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.ndarray.copy(self.cached_aats_predict[k][j])))))
    #                 self.A_predict[k].append([1/self.T*math.sqrt(((self.big_b[k][i] + self.m_h[k][i].T@self.m_h[k][i])*big_matrix).sum()) for i in range(self.D[k+1])])
    #                 rho_pr_slot = []
    #                 a_mean_prev = np.vstack((1, self.cached_means_predict[k][j]))
    #                 aat_mean_prev = np.hstack((a_mean_prev, np.vstack((a_mean_prev[1:].T, self.cached_aats_predict[k][j]))))
    #                 for i in range(self.D[k+1]):
    #                     aat_shifted = aat_mean_prev@np.hstack((self.b_predict[k][j][i], self.M_predict[k][j][i])).reshape(self.D[k] + 1,1)
    #                     eta = inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])
    #                     rho_pr_slot.append(better_sigmoid((-0.5*eta*((self.big_b[k][i]+self.m_h[k][i].T@self.m_h[k][i])*(aat_mean_prev).T).sum()\
    #                     + eta*self.m_h[k][i]@aat_shifted\
    #                     + 1/self.T*self.m_h[k][i]@a_mean_prev).item()))
    #                 self.rho_predict[k][j] = np.copy(rho_pr_slot)

    #         if ep_pred%10==0:
    #             temp_sum = 0
    #             for j in range(N_pred):
    #                 for k in range(self.L):
    #                     a_mean_prev =   self.cached_means_predict[k][j]
    #                     aat_prev = self.cached_aats_predict[k][j]
    #                     aat_current = self.cached_aats_predict[k+1][j]
    #                     a_current =  self.cached_means_predict[k+1][j]
    #                     for i in range(self.D[k+1]):
    #                         aat_shifted = self.b_predict[k][j][i]*a_mean_prev + (aat_prev@self.M_predict[k][j][i]).reshape(self.D[k],1)
    #                         big_trace = self.big_b[k][i][0,0] + self.m_h[k][i][0,0]**2 + 2*(self.big_b[k][i][0,1:] +self.m_h[k][i][0,0]*self.m_h[k][i][:,1:])@a_mean_prev\
    #                             + ((self.big_b[k][i][1:,1:] + self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*(aat_prev).T).sum()
    #                         small_trace = self.m_h[k][i][0,0]**2 + 2*self.m_h[k][i][0,0]*self.m_h[k][i][:,1:]@a_mean_prev + ((self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*(a_mean_prev@a_mean_prev.T).T).sum()
    #                         temp_sum += -0.5*inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*((self.rho_predict[k][j][i]*(self.m_h[k][i][:,1:]@a_mean_prev +self.m_h[k][i][0,0])- a_current[i])**2\
    #                                                                                         + aat_current[i,i] - a_current[i]**2 + self.rho_predict[k][j][i]*big_trace -self.rho_predict[k][j][i]**2*small_trace\
    #                                                                                             + self.rho_predict[k][j][i]*self.m_h[k][i][:,1:]@(a_current[i]*a_mean_prev - aat_shifted))\
    #                                         + 1/self.T*(self.rho_predict[k][j][i] - 0.5)*(self.m_h[k][i][:,1:]@a_mean_prev +self.m_h[k][i][0,0]) - 1/(2*self.T**2)*pg_mean(self.A_predict[k][j][i])*big_trace\
    #                                             - smart_log(self.rho_predict[k][j][i]) - smart_log((1-self.rho_predict[k][j][i])) - logcosh(self.A_predict[k][j][i]*0.5) + 0.5*pg_mean(self.A_predict[k][j][i])*self.A_predict[k][j][i]**2

    #                     temp_sum += 0.5*np.linalg.slogdet(self.S_predict[k][j])[1]   

    #             self.elbo_pred.append(temp_sum) 
        

    #         if len(self.elbo_pred)>3:
    #             if np.abs(1 - self.elbo_pred[-1]/self.elbo_pred[-2])<rate_pred:
    #                 break

        
    #     self.prediction_mean = np.array([[(self.m_o[j][:, 1:]@self.cached_means_predict[self.L][i] + self.m_o[j][0,0]).item() for j in range(self.D[self.L+1])] for i in range(N_pred)])

    #     pred_mean = []
    #     for i in range(N_pred):
    #         pred_mean.append(self.cached_means_predict[self.L][i])
        

    #     self.var_lin = np.array([[self.big_b[-1][j][0,0] + 2*self.big_b[-1][j][0,1:].reshape(1, self.D[self.L])@pred_mean[i]\
    #                             + ((self.big_b[-1][j][1:,1:]+ self.m_o[j][:, 1:].T@self.m_o[j][:, 1:])*self.cached_aats_predict[self.L][i].T).sum()\
    #                                 - self.m_o[j][:, 1:]@(pred_mean[i]@pred_mean[i].T)@self.m_o[j][:, 1:].T for i in range(N_pred)] for j in range(self.D[self.L + 1])]).reshape(self.prediction_mean.shape)
        

    #     self.var_tot = np.array([self.var_lin[:, i] + invgamma.mean(a = self.alpha_0, loc = 0, scale = self.beta_0[i]) for i in range(self.D[-1])]).reshape(self.prediction_mean.shape)

    #     print('Prediction done in', round(time.time() - pred_time_start, 2), 'seconds.')
    
    # def FDR(self, ka):
    #     total_numer = 0
    #     total_denom = 0
    #     for k in range(self.L):
    #         weights = self.m_h[k].squeeze()[:, 1:]
    #         stdevs = np.array([np.sqrt(np.diag(self.big_b[k][i][1:, 1:])) for i in range(self.D[k+1])])
    #         # Vectorized p-value computation
    #         z_scores = weights / stdevs
    #         p_values = np.maximum(1 - norm.cdf(-z_scores), norm.cdf(-z_scores))
            
    #         significant_mask = p_values > ka
    #         total_numer += np.sum((1 - p_values) * significant_mask)
    #         total_denom += np.sum(significant_mask)

    #     # Output layer
    #     weights_o = np.array([self.m_o[i][0,1:] for i in range(self.D[self.L+1])])
    #     stdevs_o = np.array([np.sqrt(np.diag(self.big_b[self.L][i][1:, 1:])) for i in range(self.D[self.L+1])])
        
    #     z_scores_o = weights_o / stdevs_o
    #     p_values_o = np.maximum(1 - norm.cdf(-z_scores_o), norm.cdf(-z_scores_o))
        
    #     significant_mask_o = p_values_o > ka
    #     total_numer += np.sum((1 - p_values_o) * significant_mask_o)
    #     total_denom += np.sum(significant_mask_o)
        
    #     return total_numer / total_denom if total_denom > 0 else 0
        
    # def sparse_weights(self, alpha=1e-4):
    #     # Vectorized p-value computation for all layers
    #     all_p_values = []
        
    #     for k in range(self.L):
    #         weights = self.m_h[k].squeeze()[:, 1:]
    #         stdevs = np.array([np.sqrt(np.diag(self.big_b[k][i][1:, 1:])) for i in range(self.D[k+1])])
    #         p_vals = np.maximum(1 - norm.cdf(-weights/stdevs), norm.cdf(-weights/stdevs))
    #         all_p_values.extend(p_vals.flatten())

    #     # Output layer p-values
    #     weights_o = np.array([self.m_o[i][0,1:] for i in range(self.D[self.L+1])])
    #     stdevs_o = np.array([np.sqrt(np.diag(self.big_b[self.L][i][1:, 1:])) for i in range(self.D[self.L+1])])
    #     p_vals_o = np.maximum(1 - norm.cdf(-weights_o/stdevs_o), norm.cdf(-weights_o/stdevs_o))
    #     all_p_values.extend(p_vals_o.flatten())

    #     # Find optimal kappa
    #     unique_kappas = sorted(list(set(all_p_values)), reverse=True)
    #     if 1.0 in unique_kappas:
    #         unique_kappas.remove(1.0)
            
    #     kappa = max(unique_kappas) if unique_kappas else 0.5
        
    #     for ka in unique_kappas:
    #         if self.FDR(ka) < alpha:
    #             kappa = ka
    #             continue
    #         else:
    #             break
        
    #     # Apply sparsity thresholding
    #     W_stars = []
    #     for k in range(self.L):
    #         weights = self.m_h[k].squeeze()[:, 1:]
    #         stdevs = np.array([np.sqrt(np.diag(self.big_b[k][i][1:, 1:])) for i in range(self.D[k+1])])
    #         p_vals = np.maximum(1 - norm.cdf(-weights/stdevs), norm.cdf(-weights/stdevs))
            
    #         # Vectorized active neuron check
    #         active_neurons = np.array([len(self.rho[k][:, i][(self.rho[k][:, i] < 0.5)]) < 0.99*self.N 
    #                                     for i in range(self.D[k+1])])
            
    #         W_star = np.where((p_vals >= kappa) & active_neurons[:, None], weights, 0)
    #         W_stars.append(W_star)

    #     # Output layer
    #     active_neurons_o = np.array([len(self.rho[self.L-1][:, j][(self.rho[self.L-1][:, j] < 0.5)]) < 0.99*self.N 
    #                                 for j in range(self.D[self.L])])
        
    #     W_star_o = np.where((p_vals_o >= kappa) & active_neurons_o[None, :], weights_o, 0)
    #     W_stars.append(W_star_o)

      
    #     for k in range(self.L):
    #         zero_outputs = np.all(W_stars[k] == 0, axis=1)
    #         zero_inputs = np.all(W_stars[k+1] == 0, axis=0) if k+1 < len(W_stars) else []
            
    #         for idx in np.where(zero_outputs)[0]:
    #             if k+1 < len(W_stars):
    #                 W_stars[k+1][:, idx] = 0
                    
    #         for idx in np.where(zero_inputs)[0]:
    #             W_stars[k][idx, :] = 0

    #     return W_stars
    
    # def list_connections(self, W_stars):
    #     listik = [(f"{k},{j}", f"{k+1},{i}") 
    #             for k in range(self.L+1) 
    #             for j in range(self.D[k]) 
    #             for i in range(self.D[k+1]) 
    #             if W_stars[k][i, j] != 0]
    #     return listik
    
    # def sparse_predict(self, x_for_pred, epochs_pred, alpha = 0.001, rate_pred = 0.00001):

    #     if self.A_predict is None:
    #         raise ValueError("A_predict must be initialized before calling prediction (via initializing VBNN_algorithm).")

        
    #     sparse_pred_time_start = time.time()
    #     N_pred = x_for_pred.shape[0]
    #     x_new = np.ndarray.copy(x_for_pred).reshape(N_pred, self.D[0], 1)
    #     self.elbo_pred = []
    #     W_sparse = self.sparse_weights(alpha)
    #     m_h = [np.array([np.hstack((self.m_h[k][i][:,0], W_sparse[k][i])) for i in range(self.D[k+1])]).reshape(self.m_h[k].shape) for k in range(self.L)]
    #     m_o = np.array([np.hstack((self.m_o[i][:,0], W_sparse[-1][i])) for i in range(self.D[self.L+1])]).reshape(self.m_o.shape) 
    #     big_b =  [np.copy(self.big_b[k]) for k in range(self.L+1)]
    #     for k in range(self.L+1):
    #         for i in range(self.D[k+1]):
    #             for j in range(self.D[k]):
    #                 if W_sparse[k][i][j] == 0:
    #                     big_b[k][i][j+1, :] = 0
    #                     big_b[k][i][:, j+1] = 0

    #     self.M_predict = []
    #     self.b_predict = []
    #     self.rho_predict = []
    #     x_init = np.ndarray.copy(x_new)
    #     for k in range(self.L):
    #         self.rho_predict.append(np.array([[better_sigmoid(( m_h[k][i][0, 0] +  m_h[k][i][:, 1:]@x_init[n]).item()/self.T) for i in range(self.D[k+1])] for n in range(N_pred)]))
    #         self.M_predict.append([( m_h[k].squeeze()[:, 1:]*self.rho_predict[k][j].reshape(self.D[k+1],1))for j in range(N_pred)])
    #         self.b_predict.append([( m_h[k].squeeze()[:,0]*self.rho_predict[k][j]).reshape(self.D[k+1], 1) for j in range(N_pred)])
    #         x_init = np.array([np.copy(self.M_predict[k][n])@np.copy(x_init[n]) + np.copy(self.b_predict[k][n]) for n in range(N_pred)]).reshape(N_pred, self.D[k+1], 1)

    #     self.S_predict = [np.array([0.01*np.eye(self.D[k+1])]*N_pred) for k in range(self.L)]


    #     self.cache_valid_predict = False
    #     self._compute_forward_pass_predict(x_new)

    #     for k in range(self.L):    
    #         self.A_predict[k] = []
    #         for j in range(N_pred):
    #             tobm = self.cached_means_predict[k][j]
    #             big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.ndarray.copy(self.cached_aats_predict[k][j])))))
    #             self.A_predict[k].append([1/self.T*math.sqrt(((self.big_b[k][i] +  m_h[k][i].T@ m_h[k][i])*big_matrix).sum()) for i in range(self.D[k+1])])


    #     for ep_pred in range(epochs_pred):        
    #         for k in reversed(range(self.L)):
    #             if k == self.L -1:
    #                 b_h_diag_predict =np.diag([1/inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for i in range(self.D[k+1])])
    #                 for j in range(N_pred):
    #                     self.S_predict[k][j]= np.ndarray.copy(b_h_diag_predict)
    #                     rho_dot_w =  m_h[k].squeeze()[:,1:]*self.rho_predict[k][j].reshape(self.D[k+1],1)
    #                     rho_dot_b = ( m_h[k].squeeze()[:,0]*self.rho_predict[k][j]).reshape(self.D[k+1],1)
    #                     self.b_predict[k][j] = np.copy(rho_dot_b)
    #                     self.M_predict[k][j] = np.copy(rho_dot_w)
    #             else:
    #                 # Step 1 - parameters for new a
    #                 b_h_diag_predict = np.diag([inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for i in range(self.D[k+1])])
    #                 for j in range(N_pred):
    #                     self.S_predict[k][j]= solve(b_h_diag_predict - self.M_predict[k+1][j].T@solve(self.S_predict[k+1][j], np.eye(self.D[k+1]), assume_a = 'pos')@self.M_predict[k+1][j]\
    #                                         + np.sum([(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho_predict[k+1][j][i] + 1/self.T**2*pg_mean(self.A_predict[k+1][j][i]))*(self.big_b[k+1][i][1:, 1:] +  m_h[k+1][i][:, 1:].T@ m_h[k+1][i][:, 1:]) for i in range(self.D[k+2])], axis = 0),  np.eye(self.D[k+1]), assume_a = 'pos')
    #                     rho_dot_w =  m_h[k].squeeze()[:,1:]*self.rho_predict[k][j].reshape(self.D[k+1],1)
    #                     rho_dot_b = ( m_h[k].squeeze()[:,0]*self.rho_predict[k][j]).reshape(self.D[k+1],1)
    #                     sum_for_b =  np.sum([-(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho_predict[k+1][j][i]\
    #                                         + 1/self.T**2*pg_mean(self.A_predict[k+1][j][i]))*(self.big_b[k+1][i][1:,:1] +  m_h[k+1][i][0,0]* m_h[k+1][i][:,1:].T) + 1/self.T*(self.rho_predict[k+1][j][i] - 0.5)* m_h[k+1][i][:,1:].T for i in range(self.D[k+2])], axis =0)
    #                     self.b_predict[k][j] = self.S_predict[k][j]@(b_h_diag_predict@rho_dot_b + self.M_predict[k+1][j].T@solve(self.S_predict[k+1][j],  np.eye(self.D[k+1]), assume_a = 'pos')@self.b_predict[k+1][j] + sum_for_b)
    #                     self.M_predict[k][j] = self.S_predict[k][j]@b_h_diag_predict@rho_dot_w 

    #         self.cache_valid_predict = False
    #         self._compute_forward_pass_predict(x_new)

    #         for k in range(self.L):    
    #             self.A_predict[k] = []
    #             for j in range(N_pred):
    #                 tobm = self.cached_means_predict[k][j]
    #                 big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.ndarray.copy(self.cached_aats_predict[k][j])))))
    #                 self.A_predict[k].append([1/self.T*math.sqrt(((self.big_b[k][i] +  m_h[k][i].T@ m_h[k][i])*big_matrix).sum()) for i in range(self.D[k+1])])
    #                 rho_pr_slot = []
    #                 a_mean_prev = np.vstack((1, self.cached_means_predict[k][j]))
    #                 aat_mean_prev = np.hstack((a_mean_prev, np.vstack((a_mean_prev[1:].T, self.cached_aats_predict[k][j]))))
    #                 for i in range(self.D[k+1]):
    #                     aat_shifted = aat_mean_prev@np.hstack((self.b_predict[k][j][i], self.M_predict[k][j][i])).reshape(self.D[k] + 1,1)
    #                     eta = inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])
    #                     rho_pr_slot.append(better_sigmoid((-0.5*eta*((self.big_b[k][i]+ m_h[k][i].T@ m_h[k][i])*(aat_mean_prev).T).sum()\
    #                     + eta* m_h[k][i]@aat_shifted\
    #                     + 1/self.T* m_h[k][i]@a_mean_prev).item()))
    #                 self.rho_predict[k][j] = np.copy(rho_pr_slot)

    #         if ep_pred%10==0:
    #             temp_sum = 0
    #             for j in range(N_pred):
    #                 for k in range(self.L):
    #                     a_mean_prev =  self.cached_means_predict[k][j]
    #                     aat_prev = self.cached_aats_predict[k][j]
    #                     aat_current = self.cached_aats_predict[k+1][j]
    #                     a_current = self.cached_means_predict[k+1][j]
    #                     for i in range(self.D[k+1]):
    #                         aat_shifted = self.b_predict[k][j][i]*a_mean_prev + (aat_prev@self.M_predict[k][j][i]).reshape(self.D[k],1)
    #                         big_trace = self.big_b[k][i][0,0] +  m_h[k][i][0,0]**2 + 2*(self.big_b[k][i][0,1:] + m_h[k][i][0,0]* m_h[k][i][:,1:])@a_mean_prev\
    #                             + ((self.big_b[k][i][1:,1:] +  m_h[k][i][:,1:].T@ m_h[k][i][:,1:])*(aat_prev).T).sum()
    #                         small_trace =  m_h[k][i][0,0]**2 + 2* m_h[k][i][0,0]* m_h[k][i][:,1:]@a_mean_prev + (( m_h[k][i][:,1:].T@ m_h[k][i][:,1:])*(a_mean_prev@a_mean_prev.T).T).sum()
    #                         temp_sum += -0.5*inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*((self.rho_predict[k][j][i]*( m_h[k][i][:,1:]@a_mean_prev + m_h[k][i][0,0])- a_current[i])**2\
    #                                                                                         + aat_current[i,i] - a_current[i]**2 + self.rho_predict[k][j][i]*big_trace -self.rho_predict[k][j][i]**2*small_trace\
    #                                                                                             + self.rho_predict[k][j][i]* m_h[k][i][:,1:]@(a_current[i]*a_mean_prev - aat_shifted))\
    #                                         + 1/self.T*(self.rho_predict[k][j][i] - 0.5)*( m_h[k][i][:,1:]@a_mean_prev + m_h[k][i][0,0]) - 1/(2*self.T**2)*pg_mean(self.A_predict[k][j][i])*big_trace\
    #                                             - smart_log(self.rho_predict[k][j][i]) - smart_log((1-self.rho_predict[k][j][i])) - logcosh(self.A_predict[k][j][i]*0.5) + 0.5*pg_mean(self.A_predict[k][j][i])*self.A_predict[k][j][i]**2

    #                     temp_sum += 0.5*np.linalg.slogdet(self.S_predict[k][j])[1]   

    #             self.elbo_pred.append(temp_sum) 
        

    #         if len(self.elbo_pred)>3:
    #             if np.abs(1 - self.elbo_pred[-2]/self.elbo_pred[-1])<rate_pred:
    #                 break

         
    #     self.sparse_prediction_mean = np.array([[(m_o[j][:, 1:]@self.cached_means_predict[self.L][i] + m_o[j][0,0]).item() for j in range(self.D[self.L+1])] for i in range(N_pred)])

    #     pred_mean = []
    #     for i in range(N_pred):
    #         pred_mean.append(self.cached_means_predict[self.L][i])
          

    #     self.sparse_var_lin = np.array([[self.big_b[-1][j][0,0] + 2*self.big_b[-1][j][0,1:].reshape(1, self.D[self.L])@pred_mean[i]\
    #                               + ((self.big_b[-1][j][1:,1:]+ m_o[j][:, 1:].T@m_o[j][:, 1:])*self.cached_aats_predict[self.L][i].T).sum()\
    #                                 - m_o[j][:, 1:]@(pred_mean[i]@pred_mean[i].T)@m_o[j][:, 1:].T for i in range(N_pred)] for j in range(self.D[self.L + 1])]).reshape(self.sparse_prediction_mean.shape)
        

    #     self.sparse_var_tot = np.array([self.sparse_var_lin[:, i] + invgamma.mean(a = self.alpha_0, loc = 0, scale = self.beta_0[i]) for i in range(self.D[-1])]).reshape(self.sparse_prediction_mean.shape)


    #     print('Sparse prediction done in', round(time.time() - sparse_pred_time_start, 2), 'seconds.')

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
 

       
