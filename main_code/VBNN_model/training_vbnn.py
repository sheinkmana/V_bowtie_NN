
import numpy as np
import math
from typing import Optional
from scipy.linalg import solve
from VBNN_model.utils import (inv_mean_IG, inv_mean_IG_eta, better_sigmoid, 
                   pg_mean)
from .mixer_vbnn import VBNNBase

from scipy.stats import invgamma
from .utils import (inv_mean_IG, inv_mean_IG_eta, better_sigmoid, 
                   pg_mean, reparametrize_nu, reparametrize_delta)


class VBNN_improving_order(VBNNBase):
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
        sample_size: 'Optional[int]' = None):

        super().__init__(
            x=x, y=y, D_H=D_H, L=L, T=T, 
            wb_mode=wb_mode, big_S=big_S, big_B=big_B, 
            beta_eta_h_prior=beta_eta_h_prior, sample_size= sample_size
        )
       
        
        self.A = [[] for _ in range(self.L) ]
        self.A_predict = [[] for _ in range(self.L) ]
        self.epoch_no = 0


    
    def update_out_tau(self):
        self.delta_glob[-1] = math.sqrt(self.delta_tau_prior**2\
        + sum([inv_mean_IG(self.nu_loc[-1][j][i], self.delta_loc[-1][j][i])*(self.m_o[j][0,i+1]**2 +  self.big_b[self.L][j][i+1, i+1])for i in range(self.D[self.L]) for j in range(self.D[self.L+1])]))
                                            
        if self.epoch_no == 0:
            self.nu_glob[-1] = self.nu_tau_prior - 0.5*self.D[self.L]*self.D[self.L+1]


    def update_out_psi(self):
        precompute = inv_mean_IG(self.nu_glob[-1], self.delta_glob[-1])
        self.delta_loc[-1] =  np.array([[np.sqrt(self.delta_psi_prior[-1]**2 + precompute*(self.m_o[j][0,i+1]**2 + self.big_b[self.L][j][i+1, i+1])) for i in range(self.D[self.L])] for j in range(self.D[self.L+1])])
    
        if self.epoch_no == 0:
            self.nu_loc[-1] =np.array([np.full(self.D[self.L],self.nu_psi_prior - 0.5)]*self.D[self.L+1])
        
        for j in range(self.D[self.L+1]):
            temp_sum = 0  
            for i in range(self.N):
                # a_mean = self.mean_finder(self.L, i)
                a_mean = self.cached_means[self.L][i]
                # aat_mean = self.aat_finder(self.L, i)
                aat_mean = self.cached_aats[self.L][i]
                temp_sum += ((self.y[i][j] - self.m_o[j][0,0] - (self.m_o[j][:, 1:]@a_mean).squeeze())**2\
                                + 2*self.big_b[self.L][j][0,1:].reshape(1, self.D[self.L])@a_mean + self.big_b[self.L][j][0,0]\
                                +((self.big_b[self.L][j][1:, 1:])*(aat_mean).T).sum()\
                                +(self.m_o[j][:, 1:]@(aat_mean - a_mean@a_mean.T)@self.m_o[j][:, 1:].T)).item()


            self.beta_0[j] = self.beta_eta_o_prior + 0.5*temp_sum

        if self.epoch_no == 0:
            self.alpha_0 = self.alpha_eta_o_prior + 0.5*self.N


    def update_out_eta(self):
        for j in range(self.D[self.L+1]):
            temp_sum = 0  
            for i in range(self.N):
                # a_mean = self.mean_finder(self.L, i)
                a_mean = self.cached_means[self.L][i]
                # aat_mean = self.aat_finder(self.L, i)
                aat_mean = self.cached_aats[self.L][i]
                temp_sum += ((self.y[i][j] - self.m_o[j][0,0] - (self.m_o[j][:, 1:]@a_mean).squeeze())**2\
                                + 2*self.big_b[self.L][j][0,1:].reshape(1, self.D[self.L])@a_mean + self.big_b[self.L][j][0,0]\
                                +((self.big_b[self.L][j][1:, 1:])*(aat_mean).T).sum()\
                                +(self.m_o[j][:, 1:]@(aat_mean - a_mean@a_mean.T)@self.m_o[j][:, 1:].T)).item()


            self.beta_0[j] = self.beta_eta_o_prior + 0.5*temp_sum

        if self.epoch_no == 0:
            self.alpha_0 = self.alpha_eta_o_prior + 0.5*self.N

    def update_out_wandb(self):
        for j in range(self.D[self.L+1]):
            temp_sum_m, temp_sum_b = 0, 0
            for i in range(self.N):
                a_tilda = np.vstack((1,  self.cached_means[self.L][i]))
                temp_sum_m += self.y[i][j]*a_tilda
                temp_sum_b += np.hstack((a_tilda, np.vstack((a_tilda[1:].T, self.cached_aats[self.L][i]))))

            self.big_b[self.L][j] = solve(np.diag([1/self.s_0**2] + [inv_mean_IG(self.nu_glob[self.L], self.delta_glob[self.L])*inv_mean_IG(self.nu_loc[self.L][j][i], self.delta_loc[self.L][j][i]) for i in range(self.D[self.L])])\
            + inv_mean_IG_eta(self.alpha_0, self.beta_0[j])*temp_sum_b, np.eye(self.D[self.L]+1), assume_a='pos')

            self.m_o[j] = inv_mean_IG_eta(self.alpha_0, self.beta_0[j])*(self.big_b[self.L][j]@temp_sum_m).T


    def update_a(self):
            b_h_diag = np.diag([inv_mean_IG_eta(self.alpha_h, self.beta_h[self.L - 1][i]) for i in range(self.D[self.L])])
            S_plug = solve(b_h_diag + np.sum([inv_mean_IG_eta(self.alpha_0, self.beta_0[i])*(self.big_b[self.L][i][1:, 1:] + self.m_o[i][:, 1:].T@self.m_o[i][:, 1:]) for i in range(self.D[self.L+1])], axis = 0), np.eye(self.D[self.L]), assume_a='pos')
            for n in range(self.N):
                self.S[self.L - 1][n] = np.copy(S_plug)
                rho_dot_w = self.m_h[self.L - 1].squeeze()[:,1:]*self.rho[self.L - 1][n].reshape(self.D[self.L],1)
                rho_dot_b = (self.m_h[self.L - 1].squeeze()[:,0]*self.rho[self.L - 1][n]).reshape(self.D[self.L],1)
                self.b[self.L - 1][n] = self.S[self.L - 1][n]@(np.sum([inv_mean_IG_eta(self.alpha_0, self.beta_0[i])*(-self.big_b[self.L][i][1:,:1] + (-self.m_o[i][0,0] + self.y[n][i])*self.m_o[i][:,1:].T) for i in range(self.D[self.L+1])], axis =0) + b_h_diag@rho_dot_b)
                self.M[self.L - 1][n] = self.S[self.L - 1][n]@b_h_diag@rho_dot_w
    
            for k in reversed(range(self.L-1)):               
                b_h_diag = np.diag([inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for i in range(self.D[k+1])])
                for n in range(self.N):
                    self.S[k][n] =solve(b_h_diag - self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+2]), assume_a='pos')@self.M[k+1][n]\
                    # self.S[k][n] =solve(b_h_diag - self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+1]), assume_a='pos')@self.M[k+1][n]\
                                                + np.sum([(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho[k+1][n][i] + 1/self.T**2*pg_mean(self.A[k+1][n][i]))*(self.big_b[k+1][i][1:, 1:] + self.m_h[k+1][i][:, 1:].T@self.m_h[k+1][i][:, 1:]) for i in range(self.D[k+2])], axis = 0),np.eye(self.D[k+1]), assume_a = 'pos')
                    rho_dot_w = self.m_h[k].squeeze()[:,1:]*self.rho[k][n].reshape(self.D[k+1],1)
                    rho_dot_b = (self.m_h[k].squeeze()[:,0]*self.rho[k][n]).reshape(self.D[k+1],1)
                    # self.b[k][n] = self.S[k][n]@(b_h_diag@rho_dot_b + self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+1]), assume_a='pos')@self.b[k+1][n]\
                    self.b[k][n] = self.S[k][n]@(b_h_diag@rho_dot_b + self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+2]), assume_a='pos')@self.b[k+1][n]\
                                                        + np.sum([-(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho[k+1][n][i] + 1/self.T**2*pg_mean(self.A[k+1][n][i]))*(self.big_b[k+1][i][1:,:1].reshape(self.D[k+1], 1)  + self.m_h[k+1][i][0,0]*self.m_h[k+1][i][:,1:].T )\
                                                        + 1/self.T*(self.rho[k+1][n][i] - 0.5)*self.m_h[k+1][i][:,1:].T for i in range(self.D[k+2])], axis =0))
                    self.M[k][n] = self.S[k][n]@b_h_diag@rho_dot_w 

                 # Invalidate and recompute cache after parameter updates
            self.cache_valid = False
            self._compute_forward_pass()

    def update_hid_tau(self, k):
        self.delta_glob[k] = math.sqrt(self.delta_tau_prior**2 + sum([inv_mean_IG(self.nu_loc[k][i][j], self.delta_loc[k][i][j])\
        *(self.big_b[k][i][1+j,1+j] + self.m_h[k][i][0,1+j]**2) for i in range(self.D[k+1]) for j in range(self.D[k])]))

        if self.epoch_no == 0:
            self.nu_glob[k] = self.nu_tau_prior - 0.5*self.D[k+1]*self.D[k]


    def update_hid_psi(self, k):
    
        precompute = inv_mean_IG(self.nu_glob[k], self.delta_glob[k])
        self.delta_loc[k] = np.array([[np.sqrt(self.delta_psi_prior[k]**2\
        + precompute*(self.big_b[k][i][1+j,1+j] + self.m_h[k][i][0,1+j]**2)) for j in range(self.D[k])] for i in range(self.D[k+1])])

        if self.epoch_no == 0:
            self.nu_loc[k] = np.array([np.full(self.D[k], self.nu_psi_prior - 0.5)]*self.D[k+1])

    def update_hid_eta(self, k):
    # Step 8 - parameters for eta for hidden layer, eta_1
        for i in range(self.D[k+1]):
            temp_sum = np.float64(0)  
            for j in range(self.N):
                a_mean_prev = self.cached_means[k][j]
                aat_mean_prev = self.cached_aats[k][j]
                # aat_mean_prev = self.aat_finder(k, j)
                a_mean = self.b[k][j][i] + self.M[k][j][i]@a_mean_prev
                aat = self.S[k][j][i,i] + self.b[k][j][i]**2 + (self.M[k][j]@aat_mean_prev@self.M[k][j].T + self.M[k][j]@a_mean_prev@self.b[k][j].T + self.b[k][j]@a_mean_prev.T@self.M[k][j].T)[i,i]
                temp_sum += (a_mean - self.rho[k][j][i]*self.m_h[k][i][0, 0] - self.rho[k][j][i]*self.m_h[k][i][:,1:]@a_mean_prev)**2 + aat - a_mean**2\
                + self.rho[k][j][i]*((self.big_b[k][i][1:,1:] + self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*aat_mean_prev.T).sum()\
                -self.rho[k][j][i]**2*((self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*(a_mean_prev@a_mean_prev.T)).sum()\
                + self.rho[k][j][i]*(self.m_h[k][i][0,0]**2*(1-self.rho[k][j][i]) +self.big_b[k][i][0,0])\
                + 2*self.rho[k][j][i]*((self.m_h[k][i][0,0]*self.m_h[k][i][:,1:] +self.big_b[k][i][0,1:])@a_mean_prev)\
                - 2*self.rho[k][j][i]**2*self.m_h[k][i][0,0]*self.m_h[k][i][:,1:]@a_mean_prev
      
            
            self.beta_h[k][i] = self.beta_eta_h_prior + 0.5*temp_sum.item()



        if self.epoch_no == 0:
            if k == self.L - 1:
                self.alpha_h = self.alpha_eta_h_prior  + 0.5*self.N
    def update_hid_omega(self, k):
        self.A[k] = []
        #with Tilda - W and b, 1 and a. 
        for j in range(self.N):
            a_mean =  self.cached_means[k][j]
            aat = self.cached_aats[k][j]
            self.A[k].append([1/self.T*math.sqrt(((self.big_b[k][i][0,0] + self.m_h[k][i][0,0]**2 + 2*(self.big_b[k][i][0,1:] +self.m_h[k][i][0,0]*self.m_h[k][i][:,1:])@a_mean\
                                                    + ((self.big_b[k][i][1:,1:] + self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*(aat).T).sum())).item()) for i in range(self.D[k+1])])
    

    def update_hid_wandb(self, k):
               
        for i in range(self.D[k+1]):
            D_inv = np.diag(np.array([1/self.s_0**2]+ [inv_mean_IG(self.nu_glob[k], self.delta_glob[k])*inv_mean_IG(self.nu_loc[k][i][j], self.delta_loc[k][i][j]) for j in range(self.D[k])]))
            temp_sum_b = 0 
            temp_sum_m = 0 
            for j in range(self.N):
                a_tilda = np.vstack((1, self.cached_means[k][j]))
                a_tildatimesa_tilda = np.hstack((a_tilda, np.vstack((a_tilda[1:].T, self.cached_aats[k][j]))))
                aat_shifted = a_tildatimesa_tilda@np.hstack((self.b[k][j][i], self.M[k][j][i])).reshape(self.D[k] + 1,1)
                temp_sum_b += (pg_mean(self.A[k][j][i])/self.T**2 + inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*self.rho[k][j][i])*a_tildatimesa_tilda
                temp_sum_m += inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*self.rho[k][j][i]*aat_shifted\
                    + 1/self.T*(self.rho[k][j][i] - 0.5)*a_tilda
            self.big_b[k][i] = solve(D_inv + temp_sum_b, np.eye(self.D[k] + 1), assume_a='pos') 
            self.m_h[k][i] = (self.big_b[k][i]@temp_sum_m).T

    def update_hid_gamma(self, k):
        for j in range(self.N):
            rho_slot = []
            a_mean_prev = np.vstack((1, self.cached_means[k][j]))
            aat_mean_prev = np.hstack((a_mean_prev, np.vstack((a_mean_prev[1:].T, self.cached_aats[k][j]))))
            for i in range(self.D[k+1]):
                aat_shifted = aat_mean_prev@np.hstack((self.b[k][j][i], self.M[k][j][i])).reshape(self.D[k] + 1,1)
                eta = inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])
                rho_slot.append(better_sigmoid((-0.5*eta*(((self.big_b[k][i] + self.m_h[k][i].T@self.m_h[k][i])*(aat_mean_prev).T).sum())\
                + eta*self.m_h[k][i]@aat_shifted\
                + 1/self.T*self.m_h[k][i]@a_mean_prev).item()))
            self.rho[k][j] = np.array(rho_slot, copy = True)

   

   
    def compute_means(self):
            #     # order of the variational updates will be decreasing with respect to the absolute value of parameter means
    #     # restriction omega needs to be updated before a and wandb hidden
    #     # means of tau and psi  \frac{-\delta^2}{2\nu + 2}
    #     # mean of eta \frac{\beta}{\alpha -1}
    #     # mean of omega pg_mean(A)  
    #     # mean of gamma  self.rho[k][j][i]
    #     #  mean of wandb hidden  self.m_h[k][i]
    #     #  mean of wandb output self.m_o[j]
    #     #  proxy for mean of a is self.b[k][j][i] + self.M[k][j][i]@a_mean_prev, where a_mean_prev is the mean of the previous layer and we consider last layers means
        # Hidden weights DEPEND on omega


        
   
        
        means_dict = {}
        for k in range(self.L):
            tau_mean = np.abs(-self.delta_glob[k]**2 / (2*self.nu_glob[k] + 2))
            means_dict[f'hid_tau_{k}'] = {'mean': tau_mean, 'update_fn': lambda k=k: self.update_hid_tau(k), 'dependencies': [], 'layer': k}
            
            psi_means = np.array([
                -self.delta_loc[k][i][j]**2 / (2*self.nu_loc[k][i][j] + 2)
                for i in range(self.D[k+1])
                for j in range(self.D[k])
            ])
            psi_mean = np.abs(psi_means).mean()
            means_dict[f'hid_psi_{k}'] = {'mean': psi_mean, 'update_fn': lambda k=k: self.update_hid_psi(k), 'dependencies': [], 'layer': k}
     
            eta_means = np.array([
                self.beta_h[k][i] / (self.alpha_h - 1)
                for i in range(self.D[k+1])
            ])
            eta_mean = np.abs(eta_means).mean()
            means_dict[f'hid_eta_{k}'] = {'mean': eta_mean, 'update_fn': lambda k=k: self.update_hid_eta(k),'dependencies': [], 'layer': k}
            
            if self.A[k]:  # Check if initialized
                omega_means = np.array([
                    pg_mean(self.A[k][j][i])
                    for j in range(len(self.A[k]))
                    for i in range(len(self.A[k][j]))
                ])
                omega_mean = np.abs(omega_means).mean()
            else:
            # A not initialized yet - use default value
                omega_mean = 0.01
            means_dict[f'hid_omega_{k}'] ={'mean': omega_mean, 'update_fn': lambda k=k: self.update_hid_omega(k),'dependencies': [], 'layer': k}
                   
            wb_means = np.abs(self.m_h[k]).mean()
            means_dict[f'hid_wandb_{k}'] = {
                'mean': wb_means,
                'update_fn': lambda k=k: self.update_hid_wandb(k),
                'dependencies': [f'hid_omega_{k}'],
                'layer': k}
            

        
            gamma_mean = np.abs(self.rho[k] - 0.5).mean()  # Distance from 0.5
            means_dict[f'hid_gamma_{k}'] = {'mean': gamma_mean, 'update_fn': lambda k=k: self.update_hid_gamma(k),'dependencies': [], 'layer': k}
 
        tau_mean_out = np.abs(-self.delta_glob[self.L]**2 / (2*self.nu_glob[self.L] + 2))
        means_dict['out_tau'] = {'mean': tau_mean_out, 'update_fn':  self.update_out_tau, 'dependencies': [], 'layer': None}
        
   
        psi_means_out = np.array([
            -self.delta_loc[self.L][j][i]**2 / (2*self.nu_loc[self.L][j][i] + 2)
            for j in range(self.D[self.L+1])
            for i in range(self.D[self.L])
        ])
        psi_mean_out = np.abs(psi_means_out).mean()
        means_dict['out_psi'] = {'mean':psi_mean_out, 'update_fn': self.update_out_psi, 'dependencies': [], 'layer': None}

        eta_means_out = np.array([
            self.beta_0[i] / (self.alpha_0 - 1)
            for i in range(self.D[self.L+1])
        ])
        eta_mean_out = np.abs(eta_means_out).mean()
        means_dict['out_eta'] ={'mean': eta_mean_out, 'update_fn': self.update_out_eta,'dependencies': [], 'layer': None}
        
 
        wb_mean_out = np.abs(self.m_o).mean()
        means_dict['out_wandb'] ={'mean':wb_mean_out,  'update_fn': self.update_out_wandb, 'dependencies': [], 'layer': None}
     
        a_proxy = 0
        for j in range(len(self.b[self.L-1])):
            a_mean_prev = self.cached_means[self.L-1][j]
            for i in range(self.D[self.L]):
                a_val = self.b[self.L-1][j][i] + self.M[self.L-1][j][i] @ a_mean_prev
                a_proxy += np.abs(a_val).sum()
        a_proxy /= (len(self.b[self.L-1]) * self.D[self.L])
        means_dict['activations'] = {
    'mean': a_proxy,
    'update_fn': self.update_a,
    'dependencies': [f'hid_omega_{k}' for k in range(self.L)],
    'layer': None
}
        return means_dict
    
    def _topological_sort(self, means_dict):
        """
        Sort parameter updates respecting dependencies.
        
        Returns:
            List of (name, info_dict) tuples
        """
        # 1. Count dependencies (in-degree)
        in_degree = {name: len(info['dependencies']) for name, info in means_dict.items()}
        
        # 2. Start with nodes that have no dependencies
        ready = [(name, means_dict[name]) for name, degree in in_degree.items() if degree == 0]
        ready.sort(key=lambda x: x[1]['mean'], reverse=True)  # Sort by mean
        
        # 3. Process nodes, updating ready queue
        result = []
        while ready:
            current_name, current_info = ready.pop(0)  # Unpack both
            result.append((current_name, current_info))  # ← FIX: Append tuple!
            
            # Decrease in-degree of dependents
            for other_name in means_dict:
                if current_name in means_dict[other_name]['dependencies']:
                    in_degree[other_name] -= 1
                    if in_degree[other_name] == 0:
                        ready.append((other_name, means_dict[other_name]))  # ← FIX: Append tuple!
            
            # Re-sort by magnitude
            ready.sort(key=lambda x: x[1]['mean'], reverse=True)
        
        # Check for cycles
        if len(result) != len(means_dict):
            unprocessed = set(means_dict.keys()) - {name for name, _ in result}
            raise ValueError(f"Circular dependencies detected! Unprocessed: {unprocessed}")
        
        return result 
    def algorithm(self, epochs=100, rate=0.000001, EM_step=False):
        self.elbo_total = []
        for epoch in range(epochs):
            # Forward pass to get activations
            self.cache_valid = False
            self._compute_forward_pass()
            
            # Compute parameter means and get update order
            means_dict = self.compute_means()

            sorted_updates = self._topological_sort(means_dict)
            print('sorted updates', sorted_updates)
            
            # Display order
            print("\nSorted update order (highest magnitude first, respecting dependencies):")
            for i, (name, info) in enumerate(sorted_updates, 1):
                deps_str = f" [after: {', '.join(info['dependencies'])}]" if info['dependencies'] else ""
                print(f"{i:2d}. {name:20s} (mean={info['mean']:.3f}){deps_str}")
            
            # Execute updates
            print("\nExecuting updates:")
            for name, info in sorted_updates:
                    info['update_fn']()
            
            
            # Sort by decreasing absolute mean (largest first)
            # sorted_updates = sorted(
            #     means_dict.items(), 
            #     key=lambda x: x[1][0],  # Sort by mean magnitude
            #     reverse=True
            # )
            
            # # Execute updates in order of decreasing magnitude
            # for update_name, (mean_val, update_fn, _) in sorted_updates:
            #     if self.epoch_no == 0 or epoch % 10 == 0:
            #         print(f"  {update_name:20s}: mean = {mean_val:.6f}")
            #     update_fn()
            
            # EM step for delta_tau_prior
            if EM_step:
                self._new_delta()
            
            # Compute ELBO
            if self.epoch_no % 20 == 0:
                self.elbo()
            
            # Check convergence
            if len(self.elbo_total) > 3:
                if np.abs(1 - self.elbo_total[-2]/self.elbo_total[-1]) < rate:
                    print(f"Converged at epoch {epoch}")
                    break
            
            self.epoch_no += 1
        
        print(f"\nTraining completed: {epoch + 1} epochs")

    # # def compute_means(self,):
    # #     pass
    # # #  need to do this 
    # #     # order of the variational updates will be decreasing with respect to the absolute value of parameter means
    # #     # restriction omega needs to be updated before a and wandb hidden
    # #     # means of tau and psi  \frac{-\delta^2}{2\nu + 2}
    # #     # mean of eta \frac{\beta}{\alpha -1}
    # #     # mean of omega pg_mean(A)  
    # #     # mean of gamma  self.rho[k][j][i]
    # #     #  mean of wandb hidden  self.m_h[k][i]
    # #     #  mean of wandb output self.m_o[j]
    # #     #  proxy for mean of a is self.b[k][j][i] + self.M[k][j][i]@a_mean_prev, where a_mean_prev is the mean of the previous layer and we consider last layers means
    # #     #  default order is 
    # #         # for k in range(self.L): 
    # #         #     self.update_hid_tau(k) 1, 5, 9, 13, ... (self.L-1)*4 +1
    # #         #     self.update_hid_psi(k) 2, 6, 10, 14, ... (self.L-1)*4 +2
    # #         #     self.update_hid_eta(k) 3, 7, 11, 15, ... (self.L-1)*4 +3
    # #         #     self.update_hid_omega(k) 4, 8, 12, 16, ... self.L*4
    # #         # self.update_out_tau() self.L*4 +1
    # #         # self.update_out_psi() self.L*4 +2
    # #         # self.update_out_eta() self.L*4 +3
    # #         # self.update_a()  (inside pass, do not separate) self.L*4 +4
    # #         # for k in range(self.L):
    # #         #     self.update_hid_wandb(k) self.L*4 +5, self.L*4 +7, self.L*4 +9, ... self.L*4 + 5 + 2*(self.L-1)
    # #         #     self.update_hid_gamma(k) self.L*4 +6, self.L*4 +8, self.L*4 +10, ... self.L*4 + 6 + 2*(self.L-1)
    # #         # self.update_out_wandb() self.L*6 + 4
 
    # def algorithm(self, epochs=100, rate = 0.000001, EM_step = False):
    #     # training_start_time = time.time()
    #     self.elbo_total = []
    #     for _ in range(epochs):
    #         self.cache_valid = False
    #         self._compute_forward_pass()
    #         for k in range(self.L): 
    #             self.update_hid_tau(k)
    #             self.update_hid_psi(k)
    #             self.update_hid_eta(k)
    #             self.update_hid_omega(k)
    #         self.update_out_tau()
    #         self.update_out_psi()
    #         self.update_out_eta()
    #         self.update_a()
    #         for k in range(self.L):
    #             self.update_hid_wandb(k)
    #             self.update_hid_gamma(k)
    #         self.update_out_wandb()
    #         if EM_step:
    #             self._new_delta()
    #         if self.epoch_no % 20 == 0:
    #             self.elbo()
    #         if len(self.elbo_total)>3:
    #             if np.abs(1 - self.elbo_total[-2]/self.elbo_total[-1])<rate:
    #                 break
    #         self.epoch_no += 1
    #     # print(f"Training done in: {round(time.time() - training_start_time, 2)} seconds.")





class VBNN_improving(VBNNBase):
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
        sample_size: 'Optional[int]' = None):

        super().__init__(
            x=x, y=y, D_H=D_H, L=L, T=T, 
            wb_mode=wb_mode, big_S=big_S, big_B=big_B, 
            beta_eta_h_prior=beta_eta_h_prior, sample_size= sample_size
        )
       
        
        self.A = [[] for _ in range(self.L) ]
        self.A_predict = [[] for _ in range(self.L) ]
        self.epoch_no = 0


    
    def update_out_part1(self):
        self.delta_glob[-1] = math.sqrt(self.delta_tau_prior**2\
        + sum([inv_mean_IG(self.nu_loc[-1][j][i], self.delta_loc[-1][j][i])*(self.m_o[j][0,i+1]**2 +  self.big_b[self.L][j][i+1, i+1])for i in range(self.D[self.L]) for j in range(self.D[self.L+1])]))
                                            
        if self.epoch_no == 0:
            self.nu_glob[-1] = self.nu_tau_prior - 0.5*self.D[self.L]*self.D[self.L+1]


        precompute = inv_mean_IG(self.nu_glob[-1], self.delta_glob[-1])
        self.delta_loc[-1] =  np.array([[np.sqrt(self.delta_psi_prior[-1]**2 + precompute*(self.m_o[j][0,i+1]**2 + self.big_b[self.L][j][i+1, i+1])) for i in range(self.D[self.L])] for j in range(self.D[self.L+1])])
    
        if self.epoch_no == 0:
            self.nu_loc[-1] =np.array([np.full(self.D[self.L],self.nu_psi_prior - 0.5)]*self.D[self.L+1])
        
        for j in range(self.D[self.L+1]):
            temp_sum = 0  
            for i in range(self.N):
                # a_mean = self.mean_finder(self.L, i)
                a_mean = self.cached_means[self.L][i]
                # aat_mean = self.aat_finder(self.L, i)
                aat_mean = self.cached_aats[self.L][i]
                temp_sum += ((self.y[i][j] - self.m_o[j][0,0] - (self.m_o[j][:, 1:]@a_mean).squeeze())**2\
                                + 2*self.big_b[self.L][j][0,1:].reshape(1, self.D[self.L])@a_mean + self.big_b[self.L][j][0,0]\
                                +((self.big_b[self.L][j][1:, 1:])*(aat_mean).T).sum()\
                                +(self.m_o[j][:, 1:]@(aat_mean - a_mean@a_mean.T)@self.m_o[j][:, 1:].T)).item()


            self.beta_0[j] = self.beta_eta_o_prior + 0.5*temp_sum

        if self.epoch_no == 0:
            self.alpha_0 = self.alpha_eta_o_prior + 0.5*self.N


    def update_out_part2(self):
        for j in range(self.D[self.L+1]):
            temp_sum_m, temp_sum_b = 0, 0
            for i in range(self.N):
                a_tilda = np.vstack((1,  self.cached_means[self.L][i]))
                temp_sum_m += self.y[i][j]*a_tilda
                temp_sum_b += np.hstack((a_tilda, np.vstack((a_tilda[1:].T, self.cached_aats[self.L][i]))))

            self.big_b[self.L][j] = solve(np.diag([1/self.s_0**2] + [inv_mean_IG(self.nu_glob[self.L], self.delta_glob[self.L])*inv_mean_IG(self.nu_loc[self.L][j][i], self.delta_loc[self.L][j][i]) for i in range(self.D[self.L])])\
            + inv_mean_IG_eta(self.alpha_0, self.beta_0[j])*temp_sum_b, np.eye(self.D[self.L]+1), assume_a='pos')

            self.m_o[j] = inv_mean_IG_eta(self.alpha_0, self.beta_0[j])*(self.big_b[self.L][j]@temp_sum_m).T


    def update_a(self):
            b_h_diag = np.diag([inv_mean_IG_eta(self.alpha_h, self.beta_h[self.L - 1][i]) for i in range(self.D[self.L])])
            S_plug = solve(b_h_diag + np.sum([inv_mean_IG_eta(self.alpha_0, self.beta_0[i])*(self.big_b[self.L][i][1:, 1:] + self.m_o[i][:, 1:].T@self.m_o[i][:, 1:]) for i in range(self.D[self.L+1])], axis = 0), np.eye(self.D[self.L]), assume_a='pos')
            for n in range(self.N):
                self.S[self.L - 1][n] = np.copy(S_plug)
                rho_dot_w = self.m_h[self.L - 1].squeeze()[:,1:]*self.rho[self.L - 1][n].reshape(self.D[self.L],1)
                rho_dot_b = (self.m_h[self.L - 1].squeeze()[:,0]*self.rho[self.L - 1][n]).reshape(self.D[self.L],1)
                self.b[self.L - 1][n] = self.S[self.L - 1][n]@(np.sum([inv_mean_IG_eta(self.alpha_0, self.beta_0[i])*(-self.big_b[self.L][i][1:,:1] + (-self.m_o[i][0,0] + self.y[n][i])*self.m_o[i][:,1:].T) for i in range(self.D[self.L+1])], axis =0) + b_h_diag@rho_dot_b)
                self.M[self.L - 1][n] = self.S[self.L - 1][n]@b_h_diag@rho_dot_w
    
            for k in reversed(range(self.L-1)):               
                b_h_diag = np.diag([inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for i in range(self.D[k+1])])
                for n in range(self.N):
                    self.S[k][n] =solve(b_h_diag - self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+2]), assume_a='pos')@self.M[k+1][n]\
                    # self.S[k][n] =solve(b_h_diag - self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+1]), assume_a='pos')@self.M[k+1][n]\
                                                + np.sum([(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho[k+1][n][i] + 1/self.T**2*pg_mean(self.A[k+1][n][i]))*(self.big_b[k+1][i][1:, 1:] + self.m_h[k+1][i][:, 1:].T@self.m_h[k+1][i][:, 1:]) for i in range(self.D[k+2])], axis = 0),np.eye(self.D[k+1]), assume_a = 'pos')
                    rho_dot_w = self.m_h[k].squeeze()[:,1:]*self.rho[k][n].reshape(self.D[k+1],1)
                    rho_dot_b = (self.m_h[k].squeeze()[:,0]*self.rho[k][n]).reshape(self.D[k+1],1)
                    # self.b[k][n] = self.S[k][n]@(b_h_diag@rho_dot_b + self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+1]), assume_a='pos')@self.b[k+1][n]\
                    self.b[k][n] = self.S[k][n]@(b_h_diag@rho_dot_b + self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+2]), assume_a='pos')@self.b[k+1][n]\
                                                        + np.sum([-(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho[k+1][n][i] + 1/self.T**2*pg_mean(self.A[k+1][n][i]))*(self.big_b[k+1][i][1:,:1].reshape(self.D[k+1], 1)  + self.m_h[k+1][i][0,0]*self.m_h[k+1][i][:,1:].T )\
                                                        + 1/self.T*(self.rho[k+1][n][i] - 0.5)*self.m_h[k+1][i][:,1:].T for i in range(self.D[k+2])], axis =0))
                    self.M[k][n] = self.S[k][n]@b_h_diag@rho_dot_w 

                 # Invalidate and recompute cache after parameter updates
            self.cache_valid = False
            self._compute_forward_pass()



    def update_hid_part1(self, k):
        self.delta_glob[k] = math.sqrt(self.delta_tau_prior**2 + sum([inv_mean_IG(self.nu_loc[k][i][j], self.delta_loc[k][i][j])\
        *(self.big_b[k][i][1+j,1+j] + self.m_h[k][i][0,1+j]**2) for i in range(self.D[k+1]) for j in range(self.D[k])]))

        if self.epoch_no == 0:
            self.nu_glob[k] = self.nu_tau_prior - 0.5*self.D[k+1]*self.D[k]

        precompute = inv_mean_IG(self.nu_glob[k], self.delta_glob[k])
        self.delta_loc[k] = np.array([[np.sqrt(self.delta_psi_prior[k]**2\
        + precompute*(self.big_b[k][i][1+j,1+j] + self.m_h[k][i][0,1+j]**2)) for j in range(self.D[k])] for i in range(self.D[k+1])])

        if self.epoch_no == 0:
            self.nu_loc[k] = np.array([np.full(self.D[k], self.nu_psi_prior - 0.5)]*self.D[k+1])


    # Step 8 - parameters for eta for hidden layer, eta_1
        for i in range(self.D[k+1]):
            temp_sum = np.float64(0)  
            for j in range(self.N):
                a_mean_prev = self.cached_means[k][j]
                aat_mean_prev = self.cached_aats[k][j]
                # aat_mean_prev = self.aat_finder(k, j)
                a_mean = self.b[k][j][i] + self.M[k][j][i]@a_mean_prev
                aat = self.S[k][j][i,i] + self.b[k][j][i]**2 + (self.M[k][j]@aat_mean_prev@self.M[k][j].T + self.M[k][j]@a_mean_prev@self.b[k][j].T + self.b[k][j]@a_mean_prev.T@self.M[k][j].T)[i,i]
                temp_sum += (a_mean - self.rho[k][j][i]*self.m_h[k][i][0, 0] - self.rho[k][j][i]*self.m_h[k][i][:,1:]@a_mean_prev)**2 + aat - a_mean**2\
                + self.rho[k][j][i]*((self.big_b[k][i][1:,1:] + self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*aat_mean_prev.T).sum()\
                -self.rho[k][j][i]**2*((self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*(a_mean_prev@a_mean_prev.T)).sum()\
                + self.rho[k][j][i]*(self.m_h[k][i][0,0]**2*(1-self.rho[k][j][i]) +self.big_b[k][i][0,0])\
                + 2*self.rho[k][j][i]*((self.m_h[k][i][0,0]*self.m_h[k][i][:,1:] +self.big_b[k][i][0,1:])@a_mean_prev)\
                - 2*self.rho[k][j][i]**2*self.m_h[k][i][0,0]*self.m_h[k][i][:,1:]@a_mean_prev
      
            
            self.beta_h[k][i] = self.beta_eta_h_prior + 0.5*temp_sum.item()



        if self.epoch_no == 0:
            if k == self.L - 1:
                self.alpha_h = self.alpha_eta_h_prior  + 0.5*self.N

        self.A[k] = []
        #with Tilda - W and b, 1 and a. 
        for j in range(self.N):
            a_mean =  self.cached_means[k][j]
            aat = self.cached_aats[k][j]
            self.A[k].append([1/self.T*math.sqrt(((self.big_b[k][i][0,0] + self.m_h[k][i][0,0]**2 + 2*(self.big_b[k][i][0,1:] +self.m_h[k][i][0,0]*self.m_h[k][i][:,1:])@a_mean\
                                                    + ((self.big_b[k][i][1:,1:] + self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*(aat).T).sum())).item()) for i in range(self.D[k+1])])
    

    def update_hid_part2(self, k):
               
        for i in range(self.D[k+1]):
            D_inv = np.diag(np.array([1/self.s_0**2]+ [inv_mean_IG(self.nu_glob[k], self.delta_glob[k])*inv_mean_IG(self.nu_loc[k][i][j], self.delta_loc[k][i][j]) for j in range(self.D[k])]))
            temp_sum_b = 0 
            temp_sum_m = 0 
            for j in range(self.N):
                a_tilda = np.vstack((1, self.cached_means[k][j]))
                a_tildatimesa_tilda = np.hstack((a_tilda, np.vstack((a_tilda[1:].T, self.cached_aats[k][j]))))
                aat_shifted = a_tildatimesa_tilda@np.hstack((self.b[k][j][i], self.M[k][j][i])).reshape(self.D[k] + 1,1)
                temp_sum_b += (pg_mean(self.A[k][j][i])/self.T**2 + inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*self.rho[k][j][i])*a_tildatimesa_tilda
                temp_sum_m += inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*self.rho[k][j][i]*aat_shifted\
                    + 1/self.T*(self.rho[k][j][i] - 0.5)*a_tilda
            self.big_b[k][i] = solve(D_inv + temp_sum_b, np.eye(self.D[k] + 1), assume_a='pos') 
            self.m_h[k][i] = (self.big_b[k][i]@temp_sum_m).T


        for j in range(self.N):
            rho_slot = []
            a_mean_prev = np.vstack((1, self.cached_means[k][j]))
            aat_mean_prev = np.hstack((a_mean_prev, np.vstack((a_mean_prev[1:].T, self.cached_aats[k][j]))))
            for i in range(self.D[k+1]):
                aat_shifted = aat_mean_prev@np.hstack((self.b[k][j][i], self.M[k][j][i])).reshape(self.D[k] + 1,1)
                eta = inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])
                rho_slot.append(better_sigmoid((-0.5*eta*(((self.big_b[k][i] + self.m_h[k][i].T@self.m_h[k][i])*(aat_mean_prev).T).sum())\
                + eta*self.m_h[k][i]@aat_shifted\
                + 1/self.T*self.m_h[k][i]@a_mean_prev).item()))
            self.rho[k][j] = np.array(rho_slot, copy = True)

   

   

 
    def algorithm(self, epochs=100, rate = 0.000001, EM_step = False):
        # training_start_time = time.time()
        self.elbo_total = []
        for _ in range(epochs):
            self.cache_valid = False
            self._compute_forward_pass()
            for k in range(self.L): 
                self.update_hid_part1(k)
            self.update_out_part1()
            self.update_a()
            for k in range(self.L): 
                self.update_hid_part2(k)
            self.update_out_part2()
            if EM_step:
                self._new_delta()
            if self.epoch_no % 20 == 0:
                self.elbo()
            if len(self.elbo_total)>3:
                if np.abs(1 - self.elbo_total[-2]/self.elbo_total[-1])<rate:
                    break
            self.epoch_no += 1
        # print(f"Training done in: {round(time.time() - training_start_time, 2)} seconds.")




class VBNN_SVI_improving(VBNNBase):
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
        sample_size: int):

        super().__init__(
            x=x, y=y, D_H=D_H, L=L, T=T, 
            wb_mode=wb_mode, big_S=big_S, big_B=big_B, 
            beta_eta_h_prior=beta_eta_h_prior, sample_size= sample_size
        )
       
        
        self.A = [[] for _ in range(self.L) ]
        self.A_predict = [[] for _ in range(self.L) ]
        self.epoch_no = 0
         
    

    def update_cavi_params(self,):
        for k in range(self.L):
            self.delta_glob[k] = math.sqrt(self.delta_tau_prior**2 + sum([inv_mean_IG(self.nu_loc[k][i][j], self.delta_loc[k][i][j])\
                *(self.big_b[k][i][1+j,1+j] + self.m_h[k][i][0,1+j]**2) for i in range(self.D[k+1]) for j in range(self.D[k])]))

            if self.epoch_no == 0:
                self.nu_glob[k] = self.nu_tau_prior - 0.5*self.D[k+1]*self.D[k]

        self.delta_glob[-1] = math.sqrt(self.delta_tau_prior**2\
            + sum([inv_mean_IG(self.nu_loc[-1][j][i], self.delta_loc[-1][j][i])*(self.m_o[j][0,i+1]**2 +  self.big_b[self.L][j][i+1, i+1])for i in range(self.D[self.L]) for j in range(self.D[self.L+1])]))                                 
        
        if self.epoch_no == 0:
            self.nu_glob[-1] = self.nu_tau_prior - 0.5*self.D[self.L]*self.D[self.L+1]
        

        for k in range(self.L):
            precompute = inv_mean_IG(self.nu_glob[k], self.delta_glob[k])
            self.delta_loc[k] = np.array([[np.sqrt(self.delta_psi_prior[k]**2\
                + precompute*(self.big_b[k][i][1+j,1+j] + self.m_h[k][i][0,1+j]**2)) for j in range(self.D[k])] for i in range(self.D[k+1])])
            
            if self.epoch_no == 0:
                self.nu_loc[k] = np.array([np.full(self.D[k], self.nu_psi_prior - 0.5)]*self.D[k+1])

        precompute = inv_mean_IG(self.nu_glob[-1], self.delta_glob[-1])
        self.delta_loc[-1] =  np.array([[np.sqrt(self.delta_psi_prior[-1]**2 + precompute*(self.m_o[j][0,i+1]**2 + self.big_b[self.L][j][i+1, i+1])) for i in range(self.D[self.L])] for j in range(self.D[self.L+1])])
        
        if self.epoch_no == 0:
            self.nu_loc[-1] =np.array([np.full(self.D[self.L],self.nu_psi_prior - 0.5)]*self.D[self.L+1])


    def find_optimal_local_params(self, local_epochs = 40, rate_local = 0.00001):
        elbo_loc = []
        if self.sample_size is None:
            raise ValueError("sample_size must be set (not None) before using it in a range.")
        for i in range(local_epochs):
            for k in range(self.L): 
                self.A[k] = []
                #with Tilda - W and b, 1 and a. 
             
                for j in range(self.sample_size):
                    a_mean =  self.cached_means[k][j]
                    aat = self.cached_aats[k][j]
                    self.A[k].append([1/self.T*math.sqrt((self.big_b[k][i][0,0] +  self.m_h[k][i][0,0]**2 + 2*(self.big_b[k][i][0,1:] + self.m_h[k][i][0,0]* self.m_h[k][i][:,1:])@a_mean\
                        + ((self.big_b[k][i][1:,1:] +  self.m_h[k][i][:,1:].T@ self.m_h[k][i][:,1:])*(aat).T).sum()).item()) for i in range(self.D[k+1])])

            b_h_diag = np.diag([inv_mean_IG_eta(self.alpha_h, self.beta_h[self.L-1][i]) for i in range(self.D[self.L])])
            S_plug = solve(b_h_diag + np.sum([inv_mean_IG_eta(self.alpha_0, self.beta_0[i])*(self.big_b[self.L][i][1:, 1:] + self.m_o[i][:, 1:].T@self.m_o[i][:, 1:]) for i in range(self.D[self.L+1])], axis = 0), np.eye(self.D[self.L]), assume_a='pos')
            for n in range(self.sample_size):
                self.S[self.L-1][n] = np.copy(S_plug)
                rho_dot_w =  self.m_h[self.L-1].squeeze()[:,1:]*self.rho[self.L-1][n].reshape(self.D[self.L],1)
                rho_dot_b = ( self.m_h[self.L-1].squeeze()[:,0]*self.rho[self.L-1][n]).reshape(self.D[self.L],1)
                self.b[self.L-1][n] = self.S[self.L-1][n]@(np.sum([inv_mean_IG_eta(self.alpha_0, self.beta_0[i])*(-self.big_b[self.L][i][1:,:1] + (-self.m_o[i][0,0] + self.y_sample[n][i])*self.m_o[i][:,1:].T) for i in range(self.D[self.L+1])], axis =0) + b_h_diag@rho_dot_b)
                self.M[self.L-1][n] = self.S[self.L-1][n]@b_h_diag@rho_dot_w

            for k in reversed(range(self.L-1)): 
                b_h_diag = np.diag([inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for i in range(self.D[k+1])])
                for n in range(self.sample_size):
                    self.S[k][n] =solve(b_h_diag - self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+2]), assume_a='pos')@self.M[k+1][n]\
                    # self.S[k][n] =solve(b_h_diag - self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+1]), assume_a='pos')@self.M[k+1][n]\
                                                + np.sum([(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho[k+1][n][i] + 1/self.T**2*pg_mean(self.A[k+1][n][i]))*(self.big_b[k+1][i][1:, 1:] +  self.m_h[k+1][i][:, 1:].T@ self.m_h[k+1][i][:, 1:]) for i in range(self.D[k+2])], axis = 0),np.eye(self.D[k+1]), assume_a = 'pos')
                    rho_dot_w =  self.m_h[k].squeeze()[:,1:]*self.rho[k][n].reshape(self.D[k+1],1)
                    rho_dot_b = ( self.m_h[k].squeeze()[:,0]*self.rho[k][n]).reshape(self.D[k+1],1)
                    self.b[k][n] = self.S[k][n]@(b_h_diag@rho_dot_b + self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+2]), assume_a='pos')@self.b[k+1][n]\
                    # self.b[k][n] = self.S[k][n]@(b_h_diag@rho_dot_b + self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+1]), assume_a='pos')@self.b[k+1][n]\
                                                        + np.sum([-(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho[k+1][n][i] + 1/self.T**2*pg_mean(self.A[k+1][n][i]))*(self.big_b[k+1][i][1:,:1].reshape(self.D[k+1], 1)  +  self.m_h[k+1][i][0,0]* self.m_h[k+1][i][:,1:].T )\
                                                        + 1/self.T*(self.rho[k+1][n][i] - 0.5)* self.m_h[k+1][i][:,1:].T for i in range(self.D[k+2])], axis =0))
                    self.M[k][n] = self.S[k][n]@b_h_diag@rho_dot_w

            self.cache_valid = False
            self._compute_forward_pass()

            for k in range(self.L): 
                for j in range(self.sample_size):
                    rho_slot = []
                    a_mean_prev = np.vstack((1, self.cached_means[k][j]))
                    aat_mean_prev = np.hstack((a_mean_prev, np.vstack((a_mean_prev[1:].T, self.cached_aats[k][j]))))
                    for i in range(self.D[k+1]):
                        aat_shifted = aat_mean_prev@np.hstack((self.b[k][j][i], self.M[k][j][i])).reshape(self.D[k] + 1,1)
                        eta = inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])
                        rho_slot.append(better_sigmoid((-0.5*eta*(((self.big_b[k][i] +  self.m_h[k][i].T@self.m_h[k][i])*(aat_mean_prev).T).sum())\
                        + eta*self.m_h[k][i]@aat_shifted\
                        + 1/self.T*self.m_h[k][i]@a_mean_prev).item()))
                    self.rho[k][j] = np.array(rho_slot, copy = True)

            elbo_loc.append(self._elbo_agamma())

            if len(elbo_loc)>3:
                if np.abs(1 - elbo_loc[-1]/elbo_loc[-2])<rate_local and np.abs(1 - elbo_loc[-2]/elbo_loc[-3])<rate_local:
                    # print('did', i+1, 'iterations for local parameters')
                    break

    def update_global_params(self, forgrate = 0.75):
        if self.sample_size is None:
            raise ValueError("sample_size must be set (not None) before using it in a range.")
        learning_rate = 1/(2 + self.epoch_no)**forgrate
        for k in range(self.L):
            for i in range(self.D[k+1]):
                temp_sum = np.float32(0)  # or np.float32(0)
                for j in range(self.sample_size):
                    a_mean_prev = self.cached_means[k][j]
                    aat_mean_prev = self.cached_aats[k][j]
                    a_mean = self.b[k][j][i] + self.M[k][j][i]@a_mean_prev
                    aat = self.S[k][j][i,i] + self.b[k][j][i]**2 + (self.M[k][j]@aat_mean_prev@self.M[k][j].T + self.M[k][j]@a_mean_prev@self.b[k][j].T + self.b[k][j]@a_mean_prev.T@self.M[k][j].T)[i,i]
                    temp_sum += (a_mean - self.rho[k][j][i]*self.m_h[k][i][0, 0] - self.rho[k][j][i]*self.m_h[k][i][:,1:]@a_mean_prev)**2 + aat - a_mean**2\
                    + self.rho[k][j][i]*((self.big_b[k][i][1:,1:] + self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*aat_mean_prev.T).sum()\
                    -self.rho[k][j][i]**2*((self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*(a_mean_prev@a_mean_prev.T)).sum()\
                    + self.rho[k][j][i]*(self.m_h[k][i][0,0]**2*(1-self.rho[k][j][i]) +self.big_b[k][i][0,0])\
                    + 2*self.rho[k][j][i]*((self.m_h[k][i][0,0]*self.m_h[k][i][:,1:] +self.big_b[k][i][0,1:])@a_mean_prev)\
                    - 2*self.rho[k][j][i]**2*self.m_h[k][i][0,0]*self.m_h[k][i][:,1:]@a_mean_prev
                # self.beta_h[k][i] *= (1- learning_rate)
                # self.beta_h[k][i] += learning_rate*(self.beta_eta_h_prior + 0.5*float(self.N/self.sample_size)*temp_sum.item())
                intermed_beta = (1- learning_rate)/(self.beta_h[k][i]) + learning_rate/(self.beta_eta_h_prior + 0.5*float(self.N/self.sample_size)*temp_sum.item())
                self.beta_h[k][i] = 1/intermed_beta


        for j in range(self.D[self.L+1]):
            temp_sum = 0  
            for i in range(self.sample_size):
                a_mean = self.cached_means[self.L][i]
                aat_mean = self.cached_aats[self.L][i]
                temp_sum += ((self.y_sample[i][j] - self.m_o[j][0,0] - (self.m_o[j][:, 1:]@a_mean).squeeze())**2\
                                + 2*self.big_b[self.L][j][0,1:].reshape(1, self.D[self.L])@a_mean + self.big_b[self.L][j][0,0]\
                                +((self.big_b[self.L][j][1:, 1:])*(aat_mean).T).sum()\
                                +(self.m_o[j][:, 1:]@(aat_mean - a_mean@a_mean.T)@self.m_o[j][:, 1:].T)).item()    
            # self.beta_0[j] *= (1- learning_rate)
            # self.beta_0[j] += learning_rate*(self.beta_eta_o_prior + 0.5*temp_sum*float(self.N/self.sample_size))
            intermed_beta= (1- learning_rate)/(self.beta_0[j]) + learning_rate/(self.beta_eta_o_prior + 0.5*temp_sum*float(self.N/self.sample_size))
            self.beta_0[j] = 1/intermed_beta


        if self.epoch_no == 0:
            self.alpha_h = self.alpha_eta_h_prior  + 0.5*self.N
            self.alpha_0 = self.alpha_eta_o_prior + 0.5*self.N


        for k in range(self.L):
            for i in range(self.D[k+1]):
                D_inv = np.diag(np.array([1/self.s_0**2]+ [inv_mean_IG(self.nu_glob[k], self.delta_glob[k])*inv_mean_IG(self.nu_loc[k][i][j], self.delta_loc[k][i][j]) for j in range(self.D[k])]))
                temp_sum_b, temp_sum_m = 0, 0
                for j in range(self.sample_size):
                    a_tilda = np.vstack((1, self.cached_means[k][j]))
                    a_tildatimesa_tilda = np.hstack((a_tilda, np.vstack((a_tilda[1:].T, self.cached_aats[k][j]))))
                    aat_shifted = a_tildatimesa_tilda@np.hstack((self.b[k][j][i], self.M[k][j][i])).reshape(self.D[k] + 1,1)
                    temp_sum_b += (pg_mean(self.A[k][j][i])/self.T**2 + inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*self.rho[k][j][i])*a_tildatimesa_tilda
                    temp_sum_m += inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*self.rho[k][j][i]*aat_shifted\
                        + 1/self.T*(self.rho[k][j][i] - 0.5)*a_tilda
                intermed_b = (1- learning_rate)*solve(self.big_b[k][i], np.eye(self.D[k] + 1), assume_a='pos') + learning_rate*(D_inv + temp_sum_b*float(self.N/self.sample_size))
                intermed_m =(1- learning_rate)*solve(self.big_b[k][i], np.eye(self.D[k] + 1), assume_a='pos')@self.m_h[k][i].T+learning_rate*float(self.N/self.sample_size)*temp_sum_m
                self.big_b[k][i] =  solve(intermed_b, np.eye(self.D[k] + 1), assume_a='pos')
                self.m_h[k][i] = (self.big_b[k][i]@intermed_m).T

        


        for j in range(self.D[self.L+1]):
            temp_sum_m, temp_sum_b = 0, 0
            for i in range(self.sample_size):
                a_tilda = np.vstack((1,  self.cached_means[self.L][i]))
                temp_sum_m += self.y_sample[i][j]*a_tilda
                temp_sum_b += np.hstack((a_tilda, np.vstack((a_tilda[1:].T, self.cached_aats[self.L][i]))))
                


            intermed_b = (1- learning_rate)*solve(self.big_b[self.L][j], np.eye(self.D[self.L]+1), assume_a='pos') + learning_rate*(np.diag([1/self.s_0**2] + [inv_mean_IG(self.nu_glob[self.L], self.delta_glob[self.L])*inv_mean_IG(self.nu_loc[self.L][j][i], self.delta_loc[self.L][j][i]) for i in range(self.D[self.L])])\
            + inv_mean_IG_eta(self.alpha_0, self.beta_0[j])*temp_sum_b*float(self.N/self.sample_size))

            intermed_m = (1- learning_rate)*solve(np.copy(self.big_b[self.L][j]), np.eye(self.D[self.L]+1), assume_a='pos')@self.m_o[j].T + learning_rate*inv_mean_IG_eta(self.alpha_0, self.beta_0[j])*float(self.N/self.sample_size)*temp_sum_m
            self.big_b[self.L][j] =  solve(intermed_b, np.eye(self.D[self.L]+1), assume_a='pos')
            # self.big_b[self.L][j] =  solve((1- learning_rate)*solve(self.big_b[self.L][j], np.eye(self.D[self.L]+1), assume_a='pos') + learning_rate*intermed_b, np.eye(self.D[self.L]+1), assume_a='pos')
            self.m_o[j] = (self.big_b[self.L][j]@intermed_m).T





    def svi_alg(self, epochs = 1,  forgrate = 0.9, EM_step = False, rate_local = 1e-4):
        self.elbo_total = []
        for _ in range(epochs):
            self.cache_valid = False
            self._compute_forward_pass()
            self.update_cavi_params()
            self.find_optimal_local_params(rate_local = rate_local)
            self.update_global_params(forgrate = forgrate)
            if EM_step:
                self._new_delta()
            if self.epoch_no%20==0:
                self.elbo()

            self.epoch_no += 1
            self._init_sample_params()


