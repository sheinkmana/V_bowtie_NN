import numpy as np 
import math 
from scipy.stats import invgamma as invgamma, norm as gaussian, uniform as uniform, bernoulli as bernoulli, gamma as gamma
from scipy.linalg import solve
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV 
import time



# some of the functions needed for the VBNN class


# inverse of the mean of the inverse gamma distribution used for tau/psi
def inv_mean_IG(nu,delta):
    assert nu < 0, "Nu should be < 0"
    return math.exp(math.log(-2*nu) - 2*math.log(delta))

# inverse of the mean of the inverse gamma distribution used for Sigma/eta
def inv_mean_IG_eta(alpha,beta):
    return math.exp(math.log(alpha) - math.log(beta))


#stable better_sigmoid
def better_sigmoid(x):
    if -x > np.log(np.finfo(type(x)).max):
        return 0.0    
    a = np.exp(-x)
    return 1.0/ (1.0 + a)

# mean of the PG rv
def pg_mean(x):
    try:
        answer = math.exp(math.log(math.exp(x)-1) - math.log(math.exp(x)+1) - math.log(2) - math.log(x))
    except OverflowError:
        answer = 0.5/x
    return answer

# more stable computation of the log of the hyperbolic cosine
def logcosh(x):
    try:
        answer = math.log(math.cosh(x))
    except OverflowError:
        answer = x - math.log(2)
    return answer 

# used in elbo 
def smart_log(x):
    if x == 0: 
        return 0
    else: 
        return x*math.log(x)
    

# needed when want to reparametrize the parameters of IG
def reparametrize_nu(nu):
    alpha = -nu
    return alpha

def reparametrize_delta(delta):
    beta = delta**2/2
    return beta




class VBNN:
    """
    Description:
        Class for Variational Bow tie Bayesian Neural Network (VBNN)
        Trains the VBNN and obtains predictive distribution, can obtain sparse weights

    Needs:
    :param x: numpy array of shape (N, D_x) - input data for training
    :param y: numpy array of shape (N, D_x) - output for training
    :param D_a: int - number of neurons in each hidden layer
    :param L: int - number of hidden layers
    :param T: float - temperature parameter, needs to be in (0, 1)
    :param wb_mode: str - initialization mode of the model: 'laplace' or 'spikeslab'
    :param big_S: float - initial value for the covariance matrix of the stochastic activations
    :param sample_size: int - number of samples used in the stochastic optimization
    """
     
    def __init__(self, x, y, D_a, L, T, wb_mode, big_S):

        linear_regr = LinearRegression()
        linear_regr.fit(x, y)
        fitted_linr = linear_regr.predict(x)
        residuals_linr = y.reshape(fitted_linr.shape) - fitted_linr
        beta = residuals_linr.var()
        delta = np.max(np.abs(linear_regr.coef_))**0.5
        self.s_0_prior = np.abs(linear_regr.intercept_.item())**0.5

        
        self.alpha_eta_h_prior, self.alpha_eta_o_prior = 2, 2
        self.beta_eta_h_prior, self.beta_eta_o_prior = 0.01, min(beta/4, 30)
        self.nu_tau_prior, self.nu_psi_prior = -1.5, -1.5

        self.T = T
        self.s_0 = np.sqrt(self.s_0_prior)
        
        self.x = np.ndarray.copy(x).reshape(x.shape[0], x.shape[1], 1)
        self.y = np.ndarray.copy(y)
        
        
        self.L = L
        self.N = self.x.shape[0]
        self.D = [self.x.shape[1]] + [D_a]*L + [self.y.shape[1]]

        self.delta_tau_prior = np.copy(delta)
        self.delta_psi_prior = np.array([delta*self.D[1]**0.5/(self.D[0]**0.5)] + [delta]*self.L)


        self.epoch_no = 0
        
        self.big_b = [np.array([0.01*np.eye(self.D[i] + 1)]*self.D[i+1]) for i in range(self.L+1)]
        self.S = [np.array([big_S*np.eye(self.D[i+1])]*self.N) for i in range(self.L)]
        self.A = [[]]*self.L
       
        self.nu_glob = np.full(self.L +1, self.nu_tau_prior).astype(np.float64)
        self.delta_glob = np.sqrt(2*(-self.nu_glob -1)*invgamma.rvs(a=reparametrize_nu(self.nu_tau_prior), loc=0, scale=reparametrize_delta(self.delta_tau_prior), size = self.L +1))
        
        self.nu_loc =[np.full((self.D[i+1], self.D[i]), self.nu_psi_prior).astype(np.float64) for i in range(self.L + 1)]
        self.delta_loc = [np.sqrt(2*(-self.nu_loc[i]-1)*invgamma.rvs(a=reparametrize_nu(self.nu_psi_prior), loc=0, scale=reparametrize_delta(self.delta_psi_prior[i]), size =(self.D[i+1], self.D[i]))) for i in range(self.L + 1)]
        
        self.alpha_h = np.copy(self.alpha_eta_h_prior)
        self.beta_h =  np.full((self.L, self.D[1]), self.beta_eta_h_prior, dtype=np.float64)
        self.alpha_0 = np.copy(self.alpha_eta_o_prior)
        self.beta_0 =  np.full(self.D[self.L+1], self.beta_eta_o_prior, dtype=np.float64)

        self.mode = wb_mode

        self.prediction_mean, self.var_lin, self.var_tot = None, None, None
        self.elbo_total, self.elbo_pred = [], []

        self.b_predict, self.M_predict, self.S_predict, self.rho_predict = None, None, None, None
        self.A_predict = [[]]*self.L

    

# mode laplace on p 37 of paper 
        if self.mode == 'laplace':
            self.M = []
            self.b = []
            self.m_h = []
            self.rho = []
            x_set = np.copy(self.x)
            for k in range(self.L):
                m_slot_h = [[]]*self.D[k+1]                
                for i in range(self.D[k+1]):
                    part_m_1 = np.random.laplace(loc = 0, scale = np.sqrt(2/self.D[k]), size =  (1, self.D[k]))
                    delta = (x_set.squeeze().max(axis=0) - x_set.squeeze().min(axis=0))*0.05                 
                    use = np.random.uniform(low = x_set.squeeze().min(axis=0) - delta, high = x_set.squeeze().max(axis=0) + delta).reshape(self.D[k], 1)
                    m_slot_h[i] = np.hstack((-part_m_1@use, part_m_1))
                    
                self.m_h.append(np.array(m_slot_h))

                self.rho.append(np.array([[better_sigmoid((m_slot_h[i][0, 0] + self.m_h[-1][i][:, 1:]@x_set[n]).item()/self.T) for i in range(self.D[k+1])] for n in range(self.N)]))
            
                self.M.append([self.m_h[-1].squeeze()[:, 1:]*self.rho[-1][j].reshape(self.D[k+1], 1) for j in range(self.N)])
                self.b.append([(self.m_h[-1].squeeze()[:,0]*self.rho[-1][j]).reshape(self.D[k+1],1) for j in range(self.N)])

                x_set = np.array([np.copy(self.M[k][n])@np.copy(x_set[n]) + np.copy(self.b[k][n]) for n in range(self.N)])
                

            x_stand = np.copy(x_set.squeeze())
            self.m_o  = [[]]*self.D[self.L+1]
            for i in range(self.D[self.L+1]):
                reg = RidgeCV().fit(x_stand, self.y[:, i])
                self.m_o[i] = np.hstack([reg.intercept_, reg.coef_]).reshape(1, self.D[self.L] + 1)   
            
            self.m_o = np.array(self.m_o)


# mode spike - slab on p 37 of paper 
        if self.mode == 'spikeslab':
            self.M = []
            self.b = []
            self.m_h = []
            self.rho = []
            x_set = np.copy(self.x)
            for k in range(self.L):
                m_slot_h = [[]]*self.D[k+1]                
                for i in range(self.D[k+1]): 
                    vector_p = bernoulli.rvs(p = 1/(1 + np.sqrt(self.D[k])), size = (1, self.D[k]))
                    part_m_1 = gaussian.rvs(loc = 0, scale = np.sqrt(2/np.sqrt(self.D[k])), size = (1, self.D[k]))*vector_p
                    delta = (x_set.squeeze().max(axis=0) - x_set.squeeze().min(axis=0))*0.05                
                    use = np.random.uniform(low = x_set.squeeze().min(axis=0) - delta, high = x_set.squeeze().max(axis=0) + delta).reshape(self.D[k], 1)
                    m_slot_h[i] = np.hstack((-part_m_1@use, part_m_1))
                    
                self.m_h.append(np.array(m_slot_h))

                self.rho.append(np.array([[better_sigmoid((m_slot_h[i][0, 0] + self.m_h[-1][i][:, 1:]@x_set[n]).item()/self.T) for i in range(self.D[k+1])] for n in range(self.N)]))
            
                self.M.append([self.m_h[-1].squeeze()[:, 1:]*self.rho[-1][j].reshape(self.D[k+1], 1) for j in range(self.N)])
                self.b.append([(self.m_h[-1].squeeze()[:,0]*self.rho[-1][j]).reshape(self.D[k+1],1) for j in range(self.N)])

                x_set = np.array([np.copy(self.M[k][n])@np.copy(x_set[n]) + np.copy(self.b[k][n]) for n in range(self.N)])
                

            x_stand = np.copy(x_set.squeeze())
            self.m_o  = [[]]*self.D[self.L+1]
            for i in range(self.D[self.L+1]):
                reg = RidgeCV().fit(x_stand, self.y[:, i])
                self.m_o[i] = np.hstack([reg.intercept_, reg.coef_]).reshape(1, self.D[self.L] + 1)   
            
            self.m_o = np.array(self.m_o)





    def mean_finder(self, layer, n):
        if layer==0:
            mean = np.copy(self.x[n])
        if layer > 0:
            mean = np.copy(self.b[layer-1][n]) + np.ndarray.copy(self.M[layer -1][n])@np.ndarray.copy(self.mean_finder(layer -1, n))
        return np.copy(mean)
    

    def aat_finder(self, layer, n):
        if layer==0:
            var = np.ndarray.copy(self.x[n])@np.ndarray.copy(self.x[n]).T
        if layer>0:
            var = np.ndarray.copy(self.S[layer-1][n]) + np.ndarray.copy(self.b[layer - 1][n])@np.ndarray.copy(self.b[layer-1][n]).T + np.ndarray.copy(self.M[layer-1][n])@np.ndarray.copy(self.aat_finder(layer-1, n))@np.ndarray.copy(self.M[layer-1][n]).T\
                + np.ndarray.copy(self.M[layer-1][n])@np.ndarray.copy(self.mean_finder(layer-1, n))@np.ndarray.copy(self.b[layer-1][n].T) + np.ndarray.copy(self.b[layer-1][n])@np.ndarray.copy(self.mean_finder(layer-1, n)).T@np.ndarray.copy(self.M[layer-1][n]).T
        return np.ndarray.copy(var)
    

    
    def update_out_part1(self):
   
        self.delta_glob[-1] = math.sqrt(self.delta_tau_prior**2\
        + sum([inv_mean_IG(self.nu_loc[-1][j][i], self.delta_loc[-1][j][i])*(self.m_o[j][0,i+1]**2 +  self.big_b[self.L][j][i+1, i+1])for i in range(self.D[self.L]) for j in range(self.D[self.L+1])]))
                                            
        if self.epoch_no == 0:
            self.nu_glob[-1] = self.nu_tau_prior - 0.5*self.D[self.L]*self.D[self.L+1]
        

        self.delta_loc[-1] =  np.array([[np.sqrt(self.delta_psi_prior[-1]**2 + inv_mean_IG(self.nu_glob[-1], self.delta_glob[-1])*(self.m_o[j][0,i+1]**2 + self.big_b[self.L][j][i+1, i+1])) for i in range(self.D[self.L])] for j in range(self.D[self.L+1])])
    
        if self.epoch_no == 0:
            self.nu_loc[-1] =np.array([np.full(self.D[self.L],self.nu_psi_prior - 0.5)]*self.D[self.L+1])


        for j in range(self.D[self.L+1]):
            temp_sum = 0  
            for i in range(self.N):
                a_mean = self.mean_finder(self.L, i)
                aat_mean = self.aat_finder(self.L, i)
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
                a_tilda = np.vstack((1,  self.mean_finder(self.L, i)))
                temp_sum_m += self.y[i][j]*a_tilda
                temp_sum_b += np.hstack((a_tilda, np.vstack((a_tilda[1:].T, self.aat_finder(self.L, i)))))

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
                    self.S[k][n] =solve(b_h_diag - self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+1]), assume_a='pos')@self.M[k+1][n]\
                                                + np.sum([(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho[k+1][n][i] + 1/self.T**2*pg_mean(self.A[k+1][n][i]))*(self.big_b[k+1][i][1:, 1:] + self.m_h[k+1][i][:, 1:].T@self.m_h[k+1][i][:, 1:]) for i in range(self.D[k+2])], axis = 0),np.eye(self.D[k+1]), assume_a = 'pos')
                    rho_dot_w = self.m_h[k].squeeze()[:,1:]*self.rho[k][n].reshape(self.D[k+1],1)
                    rho_dot_b = (self.m_h[k].squeeze()[:,0]*self.rho[k][n]).reshape(self.D[k+1],1)
                    self.b[k][n] = self.S[k][n]@(b_h_diag@rho_dot_b + self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+1]), assume_a='pos')@self.b[k+1][n]\
                                                        + np.sum([-(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i])*self.rho[k+1][n][i] + 1/self.T**2*pg_mean(self.A[k+1][n][i]))*(self.big_b[k+1][i][1:,:1].reshape(self.D[k+1], 1)  + self.m_h[k+1][i][0,0]*self.m_h[k+1][i][:,1:].T )\
                                                        + 1/self.T*(self.rho[k+1][n][i] - 0.5)*self.m_h[k+1][i][:,1:].T for i in range(self.D[k+2])], axis =0))
                    self.M[k][n] = self.S[k][n]@b_h_diag@rho_dot_w 

    def update_hid_part1(self, k):


    # Step 6 - parameters for tau_1
        self.delta_glob[k] = math.sqrt(self.delta_tau_prior**2 + sum([inv_mean_IG(self.nu_loc[k][i][j], self.delta_loc[k][i][j])\
        *(self.big_b[k][i][1+j,1+j] + self.m_h[k][i][0,1+j]**2) for i in range(self.D[k+1]) for j in range(self.D[k])]))

        if self.epoch_no == 0:
            self.nu_glob[k] = self.nu_tau_prior - 0.5*self.D[k+1]*self.D[k]

    

        self.delta_loc[k] = np.array([[np.sqrt(self.delta_psi_prior[k]**2\
        + inv_mean_IG(self.nu_glob[k], self.delta_glob[k])*(self.big_b[k][i][1+j,1+j] + self.m_h[k][i][0,1+j]**2)) for j in range(self.D[k])] for i in range(self.D[k+1])])

        if self.epoch_no == 0:
            self.nu_loc[k] = np.array([np.full(self.D[k], self.nu_psi_prior - 0.5)]*self.D[k+1])


    # Step 8 - parameters for eta for hidden layer, eta_1
        for i in range(self.D[k+1]):
            temp_sum =0
            for j in range(self.N):
                a_mean_prev = self.mean_finder(k, j)
                aat_mean_prev = self.aat_finder(k, j)
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
            a_mean =  self.mean_finder(k, j)
            aat = self.aat_finder(k, j)
            self.A[k].append([1/self.T*math.sqrt((self.big_b[k][i][0,0] + self.m_h[k][i][0,0]**2 + 2*(self.big_b[k][i][0,1:] +self.m_h[k][i][0,0]*self.m_h[k][i][:,1:])@a_mean\
                                                    + ((self.big_b[k][i][1:,1:] + self.m_h[k][i][:,1:].T@self.m_h[k][i][:,1:])*(aat).T).sum())) for i in range(self.D[k+1])])
    

    def update_hid_part2(self, k):
               
        for i in range(self.D[k+1]):
            D_inv = np.diag(np.array([1/self.s_0**2]+ [inv_mean_IG(self.nu_glob[k], self.delta_glob[k])*inv_mean_IG(self.nu_loc[k][i][j], self.delta_loc[k][i][j]) for j in range(self.D[k])]))
            temp_sum_b = 0 
            temp_sum_m = 0 
            for j in range(self.N):
                a_tilda = np.vstack((1, self.mean_finder(k, j)))
                a_tildatimesa_tilda = np.hstack((a_tilda, np.vstack((a_tilda[1:].T, self.aat_finder(k, j)))))
                aat_shifted = a_tildatimesa_tilda@np.hstack((self.b[k][j][i], self.M[k][j][i])).reshape(self.D[k] + 1,1)
                temp_sum_b += (pg_mean(self.A[k][j][i])/self.T**2 + inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*self.rho[k][j][i])*a_tildatimesa_tilda
                temp_sum_m += inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])*self.rho[k][j][i]*aat_shifted\
                    + 1/self.T*(self.rho[k][j][i] - 0.5)*a_tilda
            self.big_b[k][i] = solve(D_inv + temp_sum_b, np.eye(self.D[k] + 1), assume_a='pos') 
            self.m_h[k][i] = (self.big_b[k][i]@temp_sum_m).T


        for j in range(self.N):
            rho_slot = []
            a_mean_prev = np.vstack((1, self.mean_finder(k, j)))
            aat_mean_prev = np.hstack((a_mean_prev, np.vstack((a_mean_prev[1:].T, self.aat_finder(k, j)))))
            for i in range(self.D[k+1]):
                aat_shifted = aat_mean_prev@np.hstack((self.b[k][j][i], self.M[k][j][i])).reshape(self.D[k] + 1,1)
                eta = inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])
                rho_slot.append(better_sigmoid((-0.5*eta*(((self.big_b[k][i] + self.m_h[k][i].T@self.m_h[k][i])*(aat_mean_prev).T).sum())\
                + eta*self.m_h[k][i]@aat_shifted\
                + 1/self.T*self.m_h[k][i]@a_mean_prev).item()))
            self.rho[k][j] = np.array(rho_slot, copy = True)

    

    def AfterSAV(self, W_stars):
        for _ in range(3):
            for k in range(self.L):
                badis = []
                badis2 = []
                for j in range(self.D[k+1]):
                    if np.all(W_stars[k][j]== 0):
                        badis.append(j)
                    if np.all(W_stars[k+1][:, j]== 0):
                        badis2.append(j)

                for j in badis:
                    for i in range(self.D[k+2]):
                        W_stars[k+1][i][j] = 0
                for j in badis2:
                    for i in range(self.D[k]):
                        W_stars[k][j][i] = 0

            listik = []
            for k in range(self.L+1):
                for j in range(self.D[k]):
                    for i in range(self.D[k+1]):
                        if W_stars[k][:, j][i] != 0:
                            listik.append(((str(k) + str(j)),(str(k+1) + str(i))))

        return W_stars, listik
    




    def FDR(self, ka):
        numer, denom = 0, 0
        for k in range(self.L):
            Ws = np.copy(self.m_h[k].squeeze()[:, 1:])
            stdev = np.array([np.sqrt(np.diag(self.big_b[k][i][1:, 1:])) for i in range(self.D[k+1])])
            for i in range(self.D[k+1]):
                for j in range(self.D[k]):
                    pci = max(1-gaussian.cdf(-Ws[i][j]/stdev[i][j]), gaussian.cdf(-Ws[i][j]/stdev[i][j]))
                    numer += (1 - pci)*(pci > ka)
                    denom += 1*(pci > ka)


        # W_o = self.m_o.squeeze()[:, 1:]
        W_o =np.array([self.m_o[i][0,1:].T for i in range(self.D[self.L +1])])
        stdev_o = np.array([np.sqrt(np.diag(self.big_b[self.L][i][1:, 1:])) for i in range(self.D[self.L +1])])
        for i in range(self.D[self.L+1]):
            for j in range(self.D[self.L]):
                pci = max(1-gaussian.cdf(-W_o[i][j]/stdev_o[i][j]), gaussian.cdf(-W_o[i][j]/stdev_o[i][j]))
                numer += (1 - pci)*(pci > ka)
                denom += 1*(pci > ka)
        return numer/denom
        
  
            
    def sparse_weighs(self, alpha = 0.0001, epsilon = 0.1):
        pcis = []
        for k in range(self.L):
            weight_means = np.copy(self.m_h[k].squeeze()[:, 1:])
            stdev = np.array([np.sqrt(np.diag(self.big_b[k][i][1:, 1:])) for i in range(self.D[k+1])])
            pcis.append([max(1-gaussian.cdf(-weight_means[i][j]/stdev[i][j]), gaussian.cdf(-weight_means[i][j]/stdev[i][j])) for j in range(self.D[k]) for i in range(self.D[k+1])])

        weight_means_o =np.array([self.m_o[i][0,1:].T for i in range(self.D[self.L +1])])
        stdev_o = np.array([np.sqrt(np.diag(self.big_b[self.L][i][1:, 1:])) for i in range(self.D[self.L +1])])
        pcis.append([max(1-gaussian.cdf(-weight_means_o[i][j]/stdev_o[i][j]), gaussian.cdf(-weight_means_o[i][j]/stdev_o[i][j])) for j in range(self.D[self.L]) for i in range(self.D[self.L+1])])

        print('----')
        kappas = sum(pcis, [])
        kappas = list(set(kappas))
        kappas.remove(1.0)
        kappas.sort(reverse = True)
        kappa = max(kappas)
        for ka in kappas:
            if self.FDR(ka) < alpha:
                # print(kappa, ka, self.FDR(ka))
                kappa = ka
                continue
            else:
                print('done, kappa upon which we threshold is ', kappa)
                break
        
        W_stars = []
        for k in range(self.L):
            weight_means = np.copy(self.m_h[k].squeeze()[:, 1:])
            stdev = np.array([np.sqrt(np.diag(self.big_b[k][i][1:, 1:])) for i in range(self.D[k+1])])
            W_star= np.zeros((self.D[k+1], self.D[k]))
            for i in range(self.D[k+1]):
                if len(self.rho[k][:, i][(self.rho[k][:, i]<0.5)]) < 0.99*self.N:
                    for j in range(self.D[k]):
                        if max(1-gaussian.cdf(-weight_means[i][j]/stdev[i][j]), gaussian.cdf(-weight_means[i][j]/stdev[i][j])) >= kappa:
                            W_star[i,j] = np.copy(weight_means[i][j])
              
            W_stars.append(np.ndarray.copy(W_star))

        W_star_o = np.zeros((self.D[self.L +1], self.D[self.L]))
        for j in range(self.D[self.L]):
            if len(self.rho[self.L-1][:, j][(self.rho[self.L-1][:, j]<0.5)]) < 0.99*self.N:
                for i in range(self.D[self.L +1]):
                    if max(1-gaussian.cdf(-weight_means_o[i][j]/stdev_o[i][j]), gaussian.cdf(-weight_means_o[i][j]/stdev_o[i][j])) >= kappa:
                        W_star_o[i,j] = np.copy(weight_means_o[i][j])

        W_stars.append(W_star_o)


   

        for i in range(1, self.L + 1):
            for j in range(self.D[i]):
                for k in range(self.D[i+1]):
                    if (np.abs(W_stars[i][k,j]*W_stars[i-1][j])<epsilon*np.ones(W_stars[i-1][j].shape)).all():
                        W_stars[i][k,j] = 0
                        W_stars[i-1][j] = np.zeros(W_stars[i-1][j].shape)

        for _ in range(3):
            for k in range(self.L):
                badis = []
                badis2 = []
                for j in range(self.D[k+1]):
                    if np.all(W_stars[k][j]== 0):
                        badis.append(j)
                    if np.all(W_stars[k+1][:, j]== 0):
                        badis2.append(j)

                for j in badis:
                    for i in range(self.D[k+2]):
                        W_stars[k+1][i][j] = 0
                for j in badis2:
                    for i in range(self.D[k]):
                        W_stars[k][j][i] = 0

        listik = []
        for k in range(self.L+1):
            for j in range(self.D[k]):
                for i in range(self.D[k+1]):
                    if W_stars[k][:, j][i] != 0:
                        listik.append(((str(k) + str(j)),(str(k+1) + str(i))))
        return W_stars, listik



    def mean_finder_predict(self, x_new, layer, n):
        if layer==0:
            mean = np.ndarray.copy(x_new[n])
        if layer > 0:
            mean = np.copy(self.b_predict[layer - 1][n]) + np.ndarray.copy(self.M_predict[layer -1][n])@np.ndarray.copy(self.mean_finder_predict(x_new, layer -1, n))
        return np.ndarray.copy(mean)


    def aat_finder_predict(self, x_new, layer, n):
        if layer==0:
            aat = np.ndarray.copy(x_new)[n]@np.copy(x_new)[n].T
        if layer>0:
            aat = np.ndarray.copy(self.S_predict[layer - 1][n]) + np.copy(self.b_predict[layer - 1][n])@np.copy(self.b_predict[layer - 1][n]).T + np.ndarray.copy(self.M_predict[layer - 1][n])@np.ndarray.copy(self.aat_finder_predict(x_new, layer-1, n))@np.ndarray.copy(self.M_predict[layer - 1][n].T)\
                + np.ndarray.copy(self.M_predict[layer - 1][n])@np.ndarray.copy(self.mean_finder_predict(x_new, layer-1, n))@np.copy(self.b_predict[layer - 1][n]).T + np.copy(self.b_predict[layer - 1][n])@np.ndarray.copy(self.mean_finder_predict(x_new, layer-1, n)).T@np.ndarray.copy(self.M_predict[layer - 1][n].T)
        return np.ndarray.copy(aat)
    

    def predict(self, x_for_pred, epochs_pred, rate_pred = 0.00001):
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

        for k in range(self.L):    
            self.A_predict[k] = []
            for j in range(N_pred):
                tobm = self.mean_finder_predict(x_new, k, j)
                big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.ndarray.copy(self.aat_finder_predict(x_new, k, j))))))
                self.A_predict[k].append([1/self.T*math.sqrt(((self.big_b[k][i] + self.m_h[k][i].T@self.m_h[k][i])*big_matrix).sum()) for i in range(self.D[k+1])])


        for _ in range(epochs_pred):        
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


            for k in range(self.L):    
                self.A_predict[k] = []
                for j in range(N_pred):
                    tobm = self.mean_finder_predict(x_new, k, j)
                    big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.ndarray.copy(self.aat_finder_predict(x_new, k, j))))))
                    self.A_predict[k].append([1/self.T*math.sqrt(((self.big_b[k][i] + self.m_h[k][i].T@self.m_h[k][i])*big_matrix).sum()) for i in range(self.D[k+1])])
                    rho_pr_slot = []
                    a_mean_prev = np.vstack((1, self.mean_finder_predict(x_new, k, j)))
                    aat_mean_prev = np.hstack((a_mean_prev, np.vstack((a_mean_prev[1:].T, self.aat_finder_predict(x_new, k, j)))))
                    for i in range(self.D[k+1]):
                        aat_shifted = aat_mean_prev@np.hstack((self.b_predict[k][j][i], self.M_predict[k][j][i])).reshape(self.D[k] + 1,1)
                        eta = inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])
                        rho_pr_slot.append(better_sigmoid((-0.5*eta*((self.big_b[k][i]+self.m_h[k][i].T@self.m_h[k][i])*(aat_mean_prev).T).sum()\
                        + eta*self.m_h[k][i]@aat_shifted\
                        + 1/self.T*self.m_h[k][i]@a_mean_prev).item()))
                    self.rho_predict[k][j] = np.copy(rho_pr_slot)

            temp_sum = 0
            for j in range(N_pred):
                for k in range(self.L):
                    a_mean_prev =  self.mean_finder_predict(x_new, k, j)
                    aat_prev = self.aat_finder_predict(x_new, k, j)
                    aat_current = self.aat_finder_predict(x_new,k+1, j)
                    a_current = self.mean_finder_predict(x_new, k+1, j)
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
                if np.abs(1 - self.elbo_pred[-1]/self.elbo_pred[-2])<rate_pred and np.abs(1 - self.elbo_pred[-2]/self.elbo_pred[-3])<rate_pred and np.abs(1 - self.elbo_pred[-3]/self.elbo_pred[-4])<rate_pred:
                    break

         
        self.prediction_mean = np.array([[(self.m_o[j][:, 1:]@self.mean_finder_predict(x_new, self.L, i) + self.m_o[j][0,0]).item() for j in range(self.D[self.L+1])] for i in range(N_pred)])

        pred_mean = []
        for i in range(N_pred):
            pred_mean.append(self.mean_finder_predict(x_new, self.L, i))
          

        self.var_lin = np.array([[self.big_b[-1][j][0,0] + 2*self.big_b[-1][j][0,1:].reshape(1, self.D[self.L])@pred_mean[i]\
                                  + ((self.big_b[-1][j][1:,1:]+ self.m_o[j][:, 1:].T@self.m_o[j][:, 1:])*self.aat_finder_predict(x_new, self.L, i).T).sum()\
                                    - self.m_o[j][:, 1:]@(pred_mean[i]@pred_mean[i].T)@self.m_o[j][:, 1:].T for i in range(N_pred)] for j in range(self.D[self.L + 1])]).reshape(self.prediction_mean.shape)
        

        self.var_tot = np.array([self.var_lin[:, i] + invgamma.mean(a = self.alpha_0, loc = 0, scale = self.beta_0[i]) for i in range(self.D[-1])]).reshape(self.prediction_mean.shape)
       

        return self.prediction_mean
    
    def sparse_predict(self, x_for_pred, epochs_pred, alpha = 0.001, rate = 0.0000001, epsilon = 0.1):
        N_pred = x_for_pred.shape[0]
        x_new = np.ndarray.copy(x_for_pred).reshape(N_pred, self.D[0], 1)
        self.elbo_pred = []
        W_sparse = self.sparse_weighs(alpha, epsilon)[0]
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

        for k in range(self.L):    
            self.A_predict[k] = []
            for j in range(N_pred):
                tobm = self.mean_finder_predict(x_new, k, j)
                big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.ndarray.copy(self.aat_finder_predict(x_new, k, j))))))
                self.A_predict[k].append([1/self.T*math.sqrt(((self.big_b[k][i] +  m_h[k][i].T@ m_h[k][i])*big_matrix).sum()) for i in range(self.D[k+1])])


        for _ in range(epochs_pred):        
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


            for k in range(self.L):    
                self.A_predict[k] = []
                for j in range(N_pred):
                    tobm = self.mean_finder_predict(x_new, k, j)
                    big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.ndarray.copy(self.aat_finder_predict(x_new, k, j))))))
                    self.A_predict[k].append([1/self.T*math.sqrt(((self.big_b[k][i] +  m_h[k][i].T@ m_h[k][i])*big_matrix).sum()) for i in range(self.D[k+1])])
                    rho_pr_slot = []
                    a_mean_prev = np.vstack((1, self.mean_finder_predict(x_new, k, j)))
                    aat_mean_prev = np.hstack((a_mean_prev, np.vstack((a_mean_prev[1:].T, self.aat_finder_predict(x_new, k, j)))))
                    for i in range(self.D[k+1]):
                        aat_shifted = aat_mean_prev@np.hstack((self.b_predict[k][j][i], self.M_predict[k][j][i])).reshape(self.D[k] + 1,1)
                        eta = inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])
                        rho_pr_slot.append(better_sigmoid((-0.5*eta*((self.big_b[k][i]+ m_h[k][i].T@ m_h[k][i])*(aat_mean_prev).T).sum()\
                        + eta* m_h[k][i]@aat_shifted\
                        + 1/self.T* m_h[k][i]@a_mean_prev).item()))
                    self.rho_predict[k][j] = np.copy(rho_pr_slot)

            temp_sum = 0
            for j in range(N_pred):
                for k in range(self.L):
                    a_mean_prev =  self.mean_finder_predict(x_new, k, j)
                    aat_prev = self.aat_finder_predict(x_new, k, j)
                    aat_current = self.aat_finder_predict(x_new,k+1, j)
                    a_current = self.mean_finder_predict(x_new, k+1, j)
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
                if np.abs(1 - self.elbo_pred[-1]/self.elbo_pred[-2])<rate and np.abs(1 - self.elbo_pred[-2]/self.elbo_pred[-3])<rate and np.abs(1 - self.elbo_pred[-3]/self.elbo_pred[-4])<rate:
                    break

         
        self.prediction_mean = np.array([[(m_o[j][:, 1:]@self.mean_finder_predict(x_new, self.L, i) + m_o[j][0,0]).item() for j in range(self.D[self.L+1])] for i in range(N_pred)])

        pred_mean = []
        for i in range(N_pred):
            pred_mean.append(self.mean_finder_predict(x_new, self.L, i))
          

        self.var_lin = np.array([[self.big_b[-1][j][0,0] + 2*self.big_b[-1][j][0,1:].reshape(1, self.D[self.L])@pred_mean[i]\
                                  + ((self.big_b[-1][j][1:,1:]+ m_o[j][:, 1:].T@m_o[j][:, 1:])*self.aat_finder_predict(x_new, self.L, i).T).sum()\
                                    - m_o[j][:, 1:]@(pred_mean[i]@pred_mean[i].T)@m_o[j][:, 1:].T for i in range(N_pred)] for j in range(self.D[self.L + 1])]).reshape(self.prediction_mean.shape)
        

        self.var_tot = np.array([self.var_lin[:, i] + invgamma.mean(a = self.alpha_0, loc = 0, scale = self.beta_0[i]) for i in range(self.D[-1])]).reshape(self.prediction_mean.shape)
       

        return self.prediction_mean


   
    def elbo_tau(self):
        elbo_tau = 0.5*sum([inv_mean_IG(self.nu_glob[k], self.delta_glob[k])*(self.delta_glob[k]**2- self.delta_tau_prior**2) for k in range(self.L+1)]) \
            +2*sum([self.nu_glob[k]*math.log(self.delta_glob[k]) for k in range(self.L+1)]) - 2*(self.L+1)*self.nu_tau_prior*math.log(self.delta_tau_prior)
        return elbo_tau
    
 
    def elbo_psi(self):
        elbo_psi= 0.5*sum([inv_mean_IG(self.nu_loc[k][i][j], self.delta_loc[k][i][j])*(self.delta_loc[k][i][j]**2 - self.delta_psi_prior[k]**2) for k in range(self.L+1) for i in range(self.D[k+1]) for j in range(self.D[k])])\
            + 2*sum([self.nu_loc[k][i][j]*math.log(self.delta_loc[k][i][j]) for k in range(self.L+1) for i in range(self.D[k+1]) for j in range(self.D[k])])
        return elbo_psi
  
     
    def elbo_eta(self):
        elbo_eta = sum([(self.beta_h[k][i]- self.beta_eta_h_prior)*inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for k in range(self.L) for i in range(self.D[k+1]) ])\
            - sum([self.alpha_h*math.log(self.beta_h[k][i]) for k in range(self.L) for i in range(self.D[k+1])])\
                + sum([(self.beta_0[i]-self.beta_eta_o_prior)*(inv_mean_IG_eta(self.alpha_0, self.beta_0[i])) for i in range(self.D[self.L+1])])\
                    - sum([self.alpha_0*math.log(self.beta_0[i]) for i in range(self.D[self.L+1])])
        return elbo_eta
   
    def elbo_WB(self):
        elbo_Wb = 0.5*sum([np.linalg.slogdet(self.big_b[k][i])[1] for k in range(self.L+1) for i in range(self.D[k+1])])\
            - 0.5*sum([(self.m_h[k][i][0,0]**2 + self.big_b[k][i][0,0])/self.s_0**2 for k in range(self.L) for i in range(self.D[k+1])])\
                - 0.5*sum([(self.m_o[i][0,0]**2 + self.big_b[self.L][i][0,0])/self.s_0**2 for i in range(self.D[self.L+1])])\
                    - 0.5*sum([(self.m_h[k][i][0,1+j]**2+ self.big_b[k][i][1+j,1+j])*inv_mean_IG(self.nu_glob[k], self.delta_glob[k])*inv_mean_IG(self.nu_loc[k][i][j], self.delta_loc[k][i][j]) for k in range(self.L) for i in range(self.D[k+1])  for j in range(self.D[k])])\
                        - 0.5*sum([(self.m_o[i][0,j+1]**2 + self.big_b[self.L][i][j+1,j+1])*inv_mean_IG(self.nu_glob[self.L], self.delta_glob[self.L])*inv_mean_IG(self.nu_loc[self.L][i][j], self.delta_loc[self.L][i][j]) for i in range(self.D[self.L+1]) for j in range(self.D[self.L])])
        return elbo_Wb
   

    
    
    def elbo_agamma(self):
        temp_sum = 0
        temp_sum_out = 0
        for j in range(self.N):
            for i in range(self.D[self.L+1]):
                a_mean = self.mean_finder(self.L, j)
                aat_mean = self.aat_finder(self.L, j)
                temp_sum_out += -0.5*inv_mean_IG_eta(self.alpha_0, self.beta_0[i])*\
                    ((self.y[j][i] - self.m_o[i][0,0] -self.m_o[i][:,1:]@a_mean)**2\
                    + (self.big_b[self.L][i][1:,1:]*aat_mean).sum()\
                    + 2*self.big_b[self.L][i][0, 1:]@a_mean+ self.big_b[self.L][i][0,0]\
                    + ((self.m_o[i][:,1:].T@self.m_o[i][:,1:])*(aat_mean - a_mean@a_mean.T)).sum())

            for k in range(self.L):
                a_mean_prev =  self.mean_finder(k, j)
                aat_prev = self.aat_finder(k, j)
                aat_current = self.aat_finder(k+1, j)
                a_current = self.mean_finder(k+1, j)
             
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
        
        return elbo_agamma
    
    
    def elbo(self):
        self.elbo_total.append((self.elbo_tau() + self.elbo_psi() + self.elbo_eta() + self.elbo_WB() + self.elbo_agamma()).item())  
  
    
    
    def new_delta(self,):
        self.delta_tau_prior = np.sqrt((self.L+1)*self.nu_tau_prior/np.sum([self.nu_glob[k]/self.delta_glob[k]**2 for k in range(self.L+1)]))
 

 
    def algorithm(self, epochs=100, rate = 0.000001, EM_step = True):
        st = time.time()
        self.elbo_total = []
        for _ in range(epochs):
            for k in range(self.L): 
                self.update_hid_part1(k)
            self.update_out_part1()
            self.update_a()
            for k in range(self.L): 
                self.update_hid_part2(k)
            self.update_out_part2()
            if EM_step:
                self.new_delta()
            self.elbo()
            if len(self.elbo_total)>10:
                if np.abs(1 - self.elbo_total[-1]/self.elbo_total[-2])<rate and np.abs(1 - self.elbo_total[-2]/self.elbo_total[-3])<rate and np.abs(1 - self.elbo_total[-3]/self.elbo_total[-4])<rate:
                    break
            self.epoch_no += 1
         
        et = time.time()
        return et-st, len(self.elbo_total)
    

    
