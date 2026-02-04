
import numpy as np
import math
from typing import Optional
from scipy.linalg import solve
from VBNN_model.utils import (inv_mean_IG, inv_mean_IG_eta, better_sigmoid, 
                   pg_mean)
from .mixer_vbnn import VBNNBase

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


    def find_optimal_local_params(self, local_epochs = 40, a = True, rho = True, rate_local = 0.00001):
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

    def update_global_params(self, weights = True, weightsout = True, eta = True, forgrate = 0.75):
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


