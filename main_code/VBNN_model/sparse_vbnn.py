import numpy as np
import time
import math
from scipy.stats import norm, invgamma
from scipy.linalg import solve
from utils import better_sigmoid, pg_mean, inv_mean_IG_eta, smart_log, logcosh
from .prediction_vbnn import VBNNPredictionMixin

class VBNNSparsityMixin(VBNNPredictionMixin):
    
    """
    Mixin for sparsification logic.
    Requires VBNNCore and VBNNPredictionMixin.
    """
    def FDR(self, ka):
        total_numer = 0
        total_denom = 0
        for k in range(self.L):
            weights = self.m_h[k].squeeze()[:, 1:]
            stdevs = np.array([np.sqrt(np.diag(self.big_b[k][i][1:, 1:])) for i in range(self.D[k+1])])
            # Vectorized p-value computation
            z_scores = weights / stdevs
            p_values = np.maximum(1 - norm.cdf(-z_scores), norm.cdf(-z_scores))
            
            significant_mask = p_values > ka
            total_numer += np.sum((1 - p_values) * significant_mask)
            total_denom += np.sum(significant_mask)

        # Output layer
        weights_o = np.array([self.m_o[i][0,1:] for i in range(self.D[self.L+1])])
        stdevs_o = np.array([np.sqrt(np.diag(self.big_b[self.L][i][1:, 1:])) for i in range(self.D[self.L+1])])
        
        z_scores_o = weights_o / stdevs_o
        p_values_o = np.maximum(1 - norm.cdf(-z_scores_o), norm.cdf(-z_scores_o))
        
        significant_mask_o = p_values_o > ka
        total_numer += np.sum((1 - p_values_o) * significant_mask_o)
        total_denom += np.sum(significant_mask_o)
        
        return total_numer / total_denom if total_denom > 0 else 0
        
    def sparse_weights(self, alpha=1e-4):
        # Vectorized p-value computation for all layers
        all_p_values = []
        
        for k in range(self.L):
            weights = self.m_h[k].squeeze()[:, 1:]
            stdevs = np.array([np.sqrt(np.diag(self.big_b[k][i][1:, 1:])) for i in range(self.D[k+1])])
            p_vals = np.maximum(1 - norm.cdf(-weights/stdevs), norm.cdf(-weights/stdevs))
            all_p_values.extend(p_vals.flatten())

        # Output layer p-values
        weights_o = np.array([self.m_o[i][0,1:] for i in range(self.D[self.L+1])])
        stdevs_o = np.array([np.sqrt(np.diag(self.big_b[self.L][i][1:, 1:])) for i in range(self.D[self.L+1])])
        p_vals_o = np.maximum(1 - norm.cdf(-weights_o/stdevs_o), norm.cdf(-weights_o/stdevs_o))
        all_p_values.extend(p_vals_o.flatten())

        # Find optimal kappa
        unique_kappas = sorted(list(set(all_p_values)), reverse=True)
        if 1.0 in unique_kappas:
            unique_kappas.remove(1.0)
            
        kappa = max(unique_kappas) if unique_kappas else 0.5
        
        for ka in unique_kappas:
            if self.FDR(ka) < alpha:
                kappa = ka
                continue
            else:
                break
        
        # Apply sparsity thresholding
        W_stars = []
        for k in range(self.L):
            weights = self.m_h[k].squeeze()[:, 1:]
            stdevs = np.array([np.sqrt(np.diag(self.big_b[k][i][1:, 1:])) for i in range(self.D[k+1])])
            p_vals = np.maximum(1 - norm.cdf(-weights/stdevs), norm.cdf(-weights/stdevs))
            
            # Vectorized active neuron check
            active_neurons = np.array([len(self.rho[k][:, i][(self.rho[k][:, i] < 0.5)]) < 0.99*self.N 
                                        for i in range(self.D[k+1])])
            
            W_star = np.where((p_vals >= kappa) & active_neurons[:, None], weights, 0)
            W_stars.append(W_star)

        # Output layer
        active_neurons_o = np.array([len(self.rho[self.L-1][:, j][(self.rho[self.L-1][:, j] < 0.5)]) < 0.99*self.N 
                                    for j in range(self.D[self.L])])
        
        W_star_o = np.where((p_vals_o >= kappa) & active_neurons_o[None, :], weights_o, 0)
        W_stars.append(W_star_o)

      
        for k in range(self.L):
            zero_outputs = np.all(W_stars[k] == 0, axis=1)
            zero_inputs = np.all(W_stars[k+1] == 0, axis=0) if k+1 < len(W_stars) else []
            
            for idx in np.where(zero_outputs)[0]:
                if k+1 < len(W_stars):
                    W_stars[k+1][:, idx] = 0
                    
            for idx in np.where(zero_inputs)[0]:
                W_stars[k][idx, :] = 0

        return W_stars
    
    def list_connections(self, W_stars):
        listik = [(f"{k},{j}", f"{k+1},{i}") 
                for k in range(self.L+1) 
                for j in range(self.D[k]) 
                for i in range(self.D[k+1]) 
                if W_stars[k][i, j] != 0]
        return listik
    
    def sparse_predict(self, x_for_pred, epochs_pred, alpha = 0.001, rate_pred = 0.00001):

        if self.A_predict is None:
            raise ValueError("A_predict must be initialized before calling prediction (via initializing VBNN_algorithm).")

        
        sparse_pred_time_start = time.time()
        N_pred = x_for_pred.shape[0]
        x_new = np.ndarray.copy(x_for_pred).reshape(N_pred, self.D[0], 1)
        self.elbo_pred = []
        W_sparse = self.sparse_weights(alpha)
        m_h = [np.array([np.hstack((self.m_h[k][i][:,0], W_sparse[k][i])) for i in range(self.D[k+1])]).reshape(self.m_h[k].shape) for k in range(self.L)]
        m_o = np.array([np.hstack((self.m_o[i][:,0], W_sparse[-1][i])) for i in range(self.D[self.L+1])]).reshape(self.m_o.shape) 
        big_b =  [np.copy(self.big_b[k]) for k in range(self.L+1)]
        for k in range(self.L+1):
            for i in range(self.D[k+1]):
                for j in range(self.D[k]):
                    if W_sparse[k][i][j] == 0:
                        big_b[k][i][j+1, :] = 0
                        big_b[k][i][:, j+1] = 0

        self.M_predict = []
        self.b_predict = []
        self.rho_predict = []
        x_init = np.ndarray.copy(x_new)
        for k in range(self.L):
            self.rho_predict.append(np.array([[better_sigmoid(( m_h[k][i][0, 0] +  m_h[k][i][:, 1:]@x_init[n]).item()/self.T) for i in range(self.D[k+1])] for n in range(N_pred)]))
            self.M_predict.append([( m_h[k].squeeze()[:, 1:]*self.rho_predict[k][j].reshape(self.D[k+1],1))for j in range(N_pred)])
            self.b_predict.append([( m_h[k].squeeze()[:,0]*self.rho_predict[k][j]).reshape(self.D[k+1], 1) for j in range(N_pred)])
            x_init = np.array([np.copy(self.M_predict[k][n])@np.copy(x_init[n]) + np.copy(self.b_predict[k][n]) for n in range(N_pred)]).reshape(N_pred, self.D[k+1], 1)

        self.S_predict = [np.array([0.01*np.eye(self.D[k+1])]*N_pred) for k in range(self.L)]


        self.cache_valid_predict = False
        self._compute_forward_pass_predict(x_new)

        for k in range(self.L):    
            self.A_predict[k] = []
            for j in range(N_pred):
                tobm = self.cached_means_predict[k][j]
                big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.ndarray.copy(self.cached_aats_predict[k][j])))))
                self.A_predict[k].append([1/self.T*math.sqrt(((self.big_b[k][i] +  m_h[k][i].T@ m_h[k][i])*big_matrix).sum()) for i in range(self.D[k+1])])


        for ep_pred in range(epochs_pred):        
            for k in reversed(range(self.L)):
                if k == self.L -1:
                    b_h_diag_predict =np.diag([1/inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for i in range(self.D[k+1])])
                    for j in range(N_pred):
                        self.S_predict[k][j]= np.ndarray.copy(b_h_diag_predict)
                        rho_dot_w =  m_h[k].squeeze()[:,1:]*self.rho_predict[k][j].reshape(self.D[k+1],1)
                        rho_dot_b = ( m_h[k].squeeze()[:,0]*self.rho_predict[k][j]).reshape(self.D[k+1],1)
                        self.b_predict[k][j] = np.copy(rho_dot_b)
                        self.M_predict[k][j] = np.copy(rho_dot_w)
                else:
                    # Step 1 - parameters for new a
                    b_h_diag_predict = np.diag([inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for i in range(self.D[k+1])])
                    for j in range(N_pred):
                        self.S_predict[k][j]= solve(b_h_diag_predict - self.M_predict[k+1][j].T@solve(self.S_predict[k+1][j], np.eye(self.D[k+1]), assume_a = 'pos')@self.M_predict[k+1][j]\
                                            + np.sum([(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho_predict[k+1][j][i] + 1/self.T**2*pg_mean(self.A_predict[k+1][j][i]))*(self.big_b[k+1][i][1:, 1:] +  m_h[k+1][i][:, 1:].T@ m_h[k+1][i][:, 1:]) for i in range(self.D[k+2])], axis = 0),  np.eye(self.D[k+1]), assume_a = 'pos')
                        rho_dot_w =  m_h[k].squeeze()[:,1:]*self.rho_predict[k][j].reshape(self.D[k+1],1)
                        rho_dot_b = ( m_h[k].squeeze()[:,0]*self.rho_predict[k][j]).reshape(self.D[k+1],1)
                        sum_for_b =  np.sum([-(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho_predict[k+1][j][i]\
                                            + 1/self.T**2*pg_mean(self.A_predict[k+1][j][i]))*(self.big_b[k+1][i][1:,:1] +  m_h[k+1][i][0,0]* m_h[k+1][i][:,1:].T) + 1/self.T*(self.rho_predict[k+1][j][i] - 0.5)* m_h[k+1][i][:,1:].T for i in range(self.D[k+2])], axis =0)
                        self.b_predict[k][j] = self.S_predict[k][j]@(b_h_diag_predict@rho_dot_b + self.M_predict[k+1][j].T@solve(self.S_predict[k+1][j],  np.eye(self.D[k+1]), assume_a = 'pos')@self.b_predict[k+1][j] + sum_for_b)
                        self.M_predict[k][j] = self.S_predict[k][j]@b_h_diag_predict@rho_dot_w 

            self.cache_valid_predict = False
            self._compute_forward_pass_predict(x_new)

            for k in range(self.L):    
                self.A_predict[k] = []
                for j in range(N_pred):
                    tobm = self.cached_means_predict[k][j]
                    big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.ndarray.copy(self.cached_aats_predict[k][j])))))
                    self.A_predict[k].append([1/self.T*math.sqrt(((self.big_b[k][i] +  m_h[k][i].T@ m_h[k][i])*big_matrix).sum()) for i in range(self.D[k+1])])
                    rho_pr_slot = []
                    a_mean_prev = np.vstack((1, self.cached_means_predict[k][j]))
                    aat_mean_prev = np.hstack((a_mean_prev, np.vstack((a_mean_prev[1:].T, self.cached_aats_predict[k][j]))))
                    for i in range(self.D[k+1]):
                        aat_shifted = aat_mean_prev@np.hstack((self.b_predict[k][j][i], self.M_predict[k][j][i])).reshape(self.D[k] + 1,1)
                        eta = inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])
                        rho_pr_slot.append(better_sigmoid((-0.5*eta*((self.big_b[k][i]+ m_h[k][i].T@ m_h[k][i])*(aat_mean_prev).T).sum()\
                        + eta* m_h[k][i]@aat_shifted\
                        + 1/self.T* m_h[k][i]@a_mean_prev).item()))
                    self.rho_predict[k][j] = np.copy(rho_pr_slot)

            if ep_pred%10==0:
                temp_sum = 0
                for j in range(N_pred):
                    for k in range(self.L):
                        a_mean_prev =  self.cached_means_predict[k][j]
                        aat_prev = self.cached_aats_predict[k][j]
                        aat_current = self.cached_aats_predict[k+1][j]
                        a_current = self.cached_means_predict[k+1][j]
                        for i in range(self.D[k+1]):
                            aat_shifted = self.b_predict[k][j][i]*a_mean_prev + (aat_prev@self.M_predict[k][j][i]).reshape(self.D[k],1)
                            big_trace = self.big_b[k][i][0,0] +  m_h[k][i][0,0]**2 + 2*(self.big_b[k][i][0,1:] + m_h[k][i][0,0]* m_h[k][i][:,1:])@a_mean_prev\
                                + ((self.big_b[k][i][1:,1:] +  m_h[k][i][:,1:].T@ m_h[k][i][:,1:])*(aat_prev).T).sum()
                            small_trace =  m_h[k][i][0,0]**2 + 2* m_h[k][i][0,0]* m_h[k][i][:,1:]@a_mean_prev + (( m_h[k][i][:,1:].T@ m_h[k][i][:,1:])*(a_mean_prev@a_mean_prev.T).T).sum()
                            temp_sum += -0.5*inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*((self.rho_predict[k][j][i]*( m_h[k][i][:,1:]@a_mean_prev + m_h[k][i][0,0])- a_current[i])**2\
                                                                                            + aat_current[i,i] - a_current[i]**2 + self.rho_predict[k][j][i]*big_trace -self.rho_predict[k][j][i]**2*small_trace\
                                                                                                + self.rho_predict[k][j][i]* m_h[k][i][:,1:]@(a_current[i]*a_mean_prev - aat_shifted))\
                                            + 1/self.T*(self.rho_predict[k][j][i] - 0.5)*( m_h[k][i][:,1:]@a_mean_prev + m_h[k][i][0,0]) - 1/(2*self.T**2)*pg_mean(self.A_predict[k][j][i])*big_trace\
                                                - smart_log(self.rho_predict[k][j][i]) - smart_log((1-self.rho_predict[k][j][i])) - logcosh(self.A_predict[k][j][i]*0.5) + 0.5*pg_mean(self.A_predict[k][j][i])*self.A_predict[k][j][i]**2

                        temp_sum += 0.5*np.linalg.slogdet(self.S_predict[k][j])[1]   

                self.elbo_pred.append(temp_sum) 
        

            if len(self.elbo_pred)>3:
                if np.abs(1 - self.elbo_pred[-2]/self.elbo_pred[-1])<rate_pred:
                    break

         
        self.sparse_prediction_mean = np.array([[(m_o[j][:, 1:]@self.cached_means_predict[self.L][i] + m_o[j][0,0]).item() for j in range(self.D[self.L+1])] for i in range(N_pred)])

        pred_mean = []
        for i in range(N_pred):
            pred_mean.append(self.cached_means_predict[self.L][i])
          

        self.sparse_var_lin = np.array([[self.big_b[-1][j][0,0] + 2*self.big_b[-1][j][0,1:].reshape(1, self.D[self.L])@pred_mean[i]\
                                  + ((self.big_b[-1][j][1:,1:]+ m_o[j][:, 1:].T@m_o[j][:, 1:])*self.cached_aats_predict[self.L][i].T).sum()\
                                    - m_o[j][:, 1:]@(pred_mean[i]@pred_mean[i].T)@m_o[j][:, 1:].T for i in range(N_pred)] for j in range(self.D[self.L + 1])]).reshape(self.sparse_prediction_mean.shape)
        

        self.sparse_var_tot = np.array([self.sparse_var_lin[:, i] + invgamma.mean(a = self.alpha_0, loc = 0, scale = self.beta_0[i]) for i in range(self.D[-1])]).reshape(self.sparse_prediction_mean.shape)


        print('Sparse prediction done in', round(time.time() - sparse_pred_time_start, 2), 'seconds.')
