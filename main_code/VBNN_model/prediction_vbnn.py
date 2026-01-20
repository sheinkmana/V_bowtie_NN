import numpy as np
import time
import math
from scipy.stats import invgamma
from numpy.typing import NDArray
from scipy.linalg import solve
from utils import better_sigmoid, pg_mean, inv_mean_IG_eta, smart_log, logcosh
from .base_vbnn import VBNNCore

class VBNNPredictionMixin(VBNNCore):
    """
    Layer 2: Prediction capabilities.
    Inherits from VBNNCore.
    """
    
    def _init_prediction_caches(self):
        self.b_predict, self.M_predict, self.S_predict, self.rho_predict, self.A_predict = None, None, None, None, None
        
        self.cached_means_predict: list[NDArray] = [
            np.zeros((self.N, self.D[i], 1)) for i in range(self.L + 1)
        ]
        self.cached_aats_predict: list[NDArray] = [
            np.zeros((self.N, self.D[i], self.D[i])) for i in range(self.L + 1)
        ]
        self.cache_valid_predict = False

      
    def _compute_forward_pass_predict(self, x_new):
        if self.cache_valid_predict:
            return

        N_pred = x_new.shape[0]

        # Ensure b_predict and M_predict are initialized
        if self.b_predict is None or self.M_predict is None or self.S_predict is None:
            raise ValueError("b_predict and M_predict must be initialized before calling _compute_forward_pass_predict.")

        # Layer 0 (input)
        self.cached_means_predict[0] = x_new  # Shape: (N_pred, D[0], 1)
        self.cached_aats_predict[0] = np.array([x @ x.T for x in x_new])  # Shape: (N_pred, D[0], D[0])

        # Forward pass through hidden layers
        for layer in range(1, self.L + 1):
            means = np.zeros((N_pred, self.D[layer], 1))
            aats = np.zeros((N_pred, self.D[layer], self.D[layer]))

            for n in range(N_pred):
                prev_mean = self.cached_means_predict[layer-1][n]
                prev_aat = self.cached_aats_predict[layer-1][n]

                # Mean computation
                means[n] = self.b_predict[layer-1][n] + self.M_predict[layer-1][n] @ prev_mean 

                # Second moment computation  
                aats[n] = (self.S_predict[layer-1][n] + 
                        self.b_predict[layer-1][n] @ self.b_predict[layer-1][n].T +
                        self.M_predict[layer-1][n] @ prev_aat @ self.M_predict[layer-1][n].T +
                        self.M_predict[layer-1][n] @ prev_mean @ self.b_predict[layer-1][n].T + 
                        self.b_predict[layer-1][n] @ prev_mean.T @ self.M_predict[layer-1][n].T)

            self.cached_means_predict[layer] = means
            self.cached_aats_predict[layer] = aats

        self.cache_valid_predict = True


    def predict(self, x_for_pred, epochs_pred, rate_pred = 0.00001):
        if self.A_predict is None:
            raise ValueError("A_predict must be initialized before calling prediction (via initializing VBNN_algorithm).")

        pred_time_start = time.time()
        N_pred = x_for_pred.shape[0]
        x_new = np.ndarray.copy(x_for_pred).reshape(N_pred, self.D[0], 1)
        self.elbo_pred = []

        self.M_predict = []
        self.b_predict = []
        self.rho_predict = []
        x_init = np.ndarray.copy(x_new)
        for k in range(self.L):
            self.rho_predict.append(np.array([[better_sigmoid((self.m_h[k][i][0, 0] + self.m_h[k][i][:, 1:]@x_init[n]).item()/self.T) for i in range(self.D[k+1])] for n in range(N_pred)]))
            self.M_predict.append([(self.m_h[k].squeeze()[:, 1:]*self.rho_predict[k][j].reshape(self.D[k+1],1))for j in range(N_pred)])
            self.b_predict.append([(self.m_h[k].squeeze()[:,0]*self.rho_predict[k][j]).reshape(self.D[k+1], 1) for j in range(N_pred)])
            x_init = np.array([np.copy(self.M_predict[k][n])@np.copy(x_init[n]) + np.copy(self.b_predict[k][n]) for n in range(N_pred)]).reshape(N_pred, self.D[k+1], 1)

        self.S_predict = [np.array([0.01*np.eye(self.D[k+1])]*N_pred) for k in range(self.L)]
        
        self.cache_valid_predict = False
        self._compute_forward_pass_predict(x_new)
        
        for k in range(self.L):    
            self.A_predict[k] = []
            for j in range(N_pred):
                tobm = self.cached_means_predict[k][j]
                # tobm = self.mean_finder_predict(x_new, k, j)
                big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.ndarray.copy(self.cached_aats_predict[k][j])))))
                self.A_predict[k].append([1/self.T*math.sqrt(((self.big_b[k][i] + self.m_h[k][i].T@self.m_h[k][i])*big_matrix).sum()) for i in range(self.D[k+1])])


        for ep_pred in range(epochs_pred):

        
            for k in reversed(range(self.L)):
                if k == self.L -1:
                    b_h_diag_predict =np.diag([1/inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for i in range(self.D[k+1])])
                    for j in range(N_pred):
                        self.S_predict[k][j]= np.ndarray.copy(b_h_diag_predict)
                        rho_dot_w = self.m_h[k].squeeze()[:,1:]*self.rho_predict[k][j].reshape(self.D[k+1],1)
                        rho_dot_b = (self.m_h[k].squeeze()[:,0]*self.rho_predict[k][j]).reshape(self.D[k+1],1)
                        self.b_predict[k][j] = np.copy(rho_dot_b)
                        self.M_predict[k][j] = np.copy(rho_dot_w)
                else:
                    # Step 1 - parameters for new a
                    b_h_diag_predict = np.diag([inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for i in range(self.D[k+1])])
                    for j in range(N_pred):
                        self.S_predict[k][j]= solve(b_h_diag_predict - self.M_predict[k+1][j].T@solve(self.S_predict[k+1][j], np.eye(self.D[k+1]), assume_a = 'pos')@self.M_predict[k+1][j]\
                                            + np.sum([(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho_predict[k+1][j][i] + 1/self.T**2*pg_mean(self.A_predict[k+1][j][i]))*(self.big_b[k+1][i][1:, 1:] + self.m_h[k+1][i][:, 1:].T@self.m_h[k+1][i][:, 1:]) for i in range(self.D[k+2])], axis = 0),  np.eye(self.D[k+1]), assume_a = 'pos')
                        rho_dot_w = self.m_h[k].squeeze()[:,1:]*self.rho_predict[k][j].reshape(self.D[k+1],1)
                        rho_dot_b = (self.m_h[k].squeeze()[:,0]*self.rho_predict[k][j]).reshape(self.D[k+1],1)
                        sum_for_b =  np.sum([-(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho_predict[k+1][j][i]\
                                            + 1/self.T**2*pg_mean(self.A_predict[k+1][j][i]))*(self.big_b[k+1][i][1:,:1] + self.m_h[k+1][i][0,0]*self.m_h[k+1][i][:,1:].T) + 1/self.T*(self.rho_predict[k+1][j][i] - 0.5)*self.m_h[k+1][i][:,1:].T for i in range(self.D[k+2])], axis =0)
                        self.b_predict[k][j] = self.S_predict[k][j]@(b_h_diag_predict@rho_dot_b + self.M_predict[k+1][j].T@solve(self.S_predict[k+1][j],  np.eye(self.D[k+1]), assume_a = 'pos')@self.b_predict[k+1][j] + sum_for_b)
                        self.M_predict[k][j] = self.S_predict[k][j]@b_h_diag_predict@rho_dot_w 
            
            self.cache_valid_predict = False
            self._compute_forward_pass_predict(x_new)


            for k in range(self.L):    
                self.A_predict[k] = []
                for j in range(N_pred):
                    tobm =  self.cached_means_predict[k][j]
                    big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.ndarray.copy(self.cached_aats_predict[k][j])))))
                    self.A_predict[k].append([1/self.T*math.sqrt(((self.big_b[k][i] + self.m_h[k][i].T@self.m_h[k][i])*big_matrix).sum()) for i in range(self.D[k+1])])
                    rho_pr_slot = []
                    a_mean_prev = np.vstack((1, self.cached_means_predict[k][j]))
                    aat_mean_prev = np.hstack((a_mean_prev, np.vstack((a_mean_prev[1:].T, self.cached_aats_predict[k][j]))))
                    for i in range(self.D[k+1]):
                        aat_shifted = aat_mean_prev@np.hstack((self.b_predict[k][j][i], self.M_predict[k][j][i])).reshape(self.D[k] + 1,1)
                        eta = inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])
                        rho_pr_slot.append(better_sigmoid((-0.5*eta*((self.big_b[k][i]+self.m_h[k][i].T@self.m_h[k][i])*(aat_mean_prev).T).sum()\
                        + eta*self.m_h[k][i]@aat_shifted\
                        + 1/self.T*self.m_h[k][i]@a_mean_prev).item()))
                    self.rho_predict[k][j] = np.copy(rho_pr_slot)

            if ep_pred%10==0:
                temp_sum = 0
                for j in range(N_pred):
                    for k in range(self.L):
                        a_mean_prev =   self.cached_means_predict[k][j]
                        aat_prev = self.cached_aats_predict[k][j]
                        aat_current = self.cached_aats_predict[k+1][j]
                        a_current =  self.cached_means_predict[k+1][j]
                        for i in range(self.D[k+1]):
                            aat_shifted = self.b_predict[k][j][i]*a_mean_prev + (aat_prev@self.M_predict[k][j][i]).reshape(self.D[k],1)
                            big_trace = self.big_b[k][i][0,0] + self.m_h[k][i][0,0]**2 + 2*(self.big_b[k][i][0,1:] +self.m_h[k][i][0,0]*self.m_h[k][i][:,1:])@a_mean_prev\
                                + ((self.big_b[k][i][1:,1:] + self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*(aat_prev).T).sum()
                            small_trace = self.m_h[k][i][0,0]**2 + 2*self.m_h[k][i][0,0]*self.m_h[k][i][:,1:]@a_mean_prev + ((self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*(a_mean_prev@a_mean_prev.T).T).sum()
                            temp_sum += -0.5*inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*((self.rho_predict[k][j][i]*(self.m_h[k][i][:,1:]@a_mean_prev +self.m_h[k][i][0,0])- a_current[i])**2\
                                                                                            + aat_current[i,i] - a_current[i]**2 + self.rho_predict[k][j][i]*big_trace -self.rho_predict[k][j][i]**2*small_trace\
                                                                                                + self.rho_predict[k][j][i]*self.m_h[k][i][:,1:]@(a_current[i]*a_mean_prev - aat_shifted))\
                                            + 1/self.T*(self.rho_predict[k][j][i] - 0.5)*(self.m_h[k][i][:,1:]@a_mean_prev +self.m_h[k][i][0,0]) - 1/(2*self.T**2)*pg_mean(self.A_predict[k][j][i])*big_trace\
                                                - smart_log(self.rho_predict[k][j][i]) - smart_log((1-self.rho_predict[k][j][i])) - logcosh(self.A_predict[k][j][i]*0.5) + 0.5*pg_mean(self.A_predict[k][j][i])*self.A_predict[k][j][i]**2

                        temp_sum += 0.5*np.linalg.slogdet(self.S_predict[k][j])[1]   

                self.elbo_pred.append(temp_sum) 
        

            if len(self.elbo_pred)>3:
                if np.abs(1 - self.elbo_pred[-1]/self.elbo_pred[-2])<rate_pred:
                    break

        
        self.prediction_mean = np.array([[(self.m_o[j][:, 1:]@self.cached_means_predict[self.L][i] + self.m_o[j][0,0]).item() for j in range(self.D[self.L+1])] for i in range(N_pred)])

        pred_mean = []
        for i in range(N_pred):
            pred_mean.append(self.cached_means_predict[self.L][i])
        

        self.var_lin = np.array([[self.big_b[-1][j][0,0] + 2*self.big_b[-1][j][0,1:].reshape(1, self.D[self.L])@pred_mean[i]\
                                + ((self.big_b[-1][j][1:,1:]+ self.m_o[j][:, 1:].T@self.m_o[j][:, 1:])*self.cached_aats_predict[self.L][i].T).sum()\
                                    - self.m_o[j][:, 1:]@(pred_mean[i]@pred_mean[i].T)@self.m_o[j][:, 1:].T for i in range(N_pred)] for j in range(self.D[self.L + 1])]).reshape(self.prediction_mean.shape)
        

        self.var_tot = np.array([self.var_lin[:, i] + invgamma.mean(a = self.alpha_0, loc = 0, scale = self.beta_0[i]) for i in range(self.D[-1])]).reshape(self.prediction_mean.shape)

        print('Prediction done in', round(time.time() - pred_time_start, 2), 'seconds.')
    