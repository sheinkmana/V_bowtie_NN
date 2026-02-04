import numpy as np
import math
from typing import Optional, List, Union
from scipy.linalg import solve

# Assuming these imports from your existing code structure
from ..utils import (inv_mean_IG, inv_mean_IG_eta, better_sigmoid, 
                   pg_mean, logcosh, smart_log)
from ..training_vbnn import VBNN_improving
from .prediction_sparse_vbnn import VBNNSparsePredictionMixin


class VBNN_SparseTraining(VBNNSparsePredictionMixin):
    """
    VBNN training with fixed sparsity pattern.
    
    Continues training only active weights/neurons identified through FDR-based
    sparsification. Inherits training logic from VBNN_improving but operates
    on pruned architecture.
    """

    # Sparsity information
    is_sparse: bool
    active_neurons: List[np.ndarray]
    active_weights: List[np.ndarray]
    W_stars: List[np.ndarray]
    
    # Architecture
    D: List[int]
    D_full: List[int]
    D_in: int
    D_out: int
    L: int
    
    # Data
    x: np.ndarray  # Training data (pruned to active features) - always set
    x_full: np.ndarray  # Original full training data
    y: np.ndarray  # Training targets - always set
    N: int

    
    # Model parameters (sparse)
    m_h: List[np.ndarray]
    m_o: np.ndarray
    big_b: List[np.ndarray]
    
    # Model parameters (full - Option 3B)
    m_h_full: List[np.ndarray]
    m_o_full: np.ndarray
    big_b_full: List[np.ndarray]
    
    # Shrinkage parameters
    nu_glob: np.ndarray
    delta_glob: np.ndarray
    nu_loc: List[np.ndarray]
    delta_loc: List[np.ndarray]
    
    # Noise parameters
    alpha_h: Union[float, np.floating]
    alpha_0: Union[float, np.floating]
    beta_h: List[np.ndarray]  # List format for variable layer sizes
    beta_h_full: Union[np.ndarray, List[np.ndarray]]
    beta_0: np.ndarray
    
    # Hyperparameters
    T: float
    mode: str
    s_0: Union[float, np.floating]
    beta_eta_h_prior: float
    beta_eta_o_prior: float
    alpha_eta_h_prior: float
    alpha_eta_o_prior: float
    nu_tau_prior: float
    nu_psi_prior: float
    delta_tau_prior: Union[float, np.floating]  # Can be float or np.float64
    delta_psi_prior: np.ndarray
    
    # Training structures
    M: List[List[np.ndarray]]
    b: List[List[np.ndarray]]
    S: List[np.ndarray]
    rho: List[np.ndarray]
    A: List[List[List[float]]]
    
    # Cached activations
    cached_means: List[np.ndarray]
    cached_aats: List[np.ndarray]
    cache_valid: bool
    
    # Prediction structures
    M_predict: Optional[List[List[np.ndarray]]]
    b_predict: Optional[List[List[np.ndarray]]]
    S_predict: Optional[List[np.ndarray]]
    rho_predict: Optional[List[np.ndarray]]
    A_predict: Optional[List[List[List[float]]]]
    cached_means_predict: Optional[List[np.ndarray]]
    cached_aats_predict: Optional[List[np.ndarray]]
    cache_valid_predict: bool
    
    # Training tracking
    epoch_no: int
    elbo_total: List[float]
    
    def __init__(self):
        """
        Direct instantiation not supported.
        
        Use VBNN_SparseTraining.from_pretrained() instead.
        
        This __init__ exists only for type checking purposes.
        """
        raise NotImplementedError(
            "VBNN_SparseTraining cannot be instantiated directly. "
            "Use VBNN_SparseTraining.from_pretrained(full_model, alpha) instead."
        )
    
    @classmethod
    def from_pretrained(
        cls,
        full_model: 'VBNN_improving',
        alpha: float = 0.01) -> 'VBNN_SparseTraining':
        """
        Create sparse training model from pretrained full model.
        
        Args:
            full_model: Trained VBNN_improving instance
            alpha: FDR threshold for sparsification (lower = more aggressive pruning)
        
        Returns:
            New VBNN_SparseTraining instance with pruned architecture
        """

        # Step 1: Run sparsification on trained model

        active_weights, active_neurons, W_stars = full_model.sparse_weights(alpha)
        

        total_params_full = sum(full_model.D[i] * full_model.D[i+1] 
                                for i in range(full_model.L + 1))
        total_params_sparse = sum(len(active_neurons[i]) * len(active_neurons[i+1]) 
                                    for i in range(full_model.L + 1))
        compression_ratio = 1 - total_params_sparse / total_params_full
        print(f"  Total parameters: {total_params_full} → {total_params_sparse}")
        print(f"  Compression: {compression_ratio*100:.1f}%")
        print("\n  Layer-wise compression:")
        for i in range(len(active_neurons)):
            if i == 0:
                layer_name = "Input"
            elif i <= full_model.L:
                layer_name = f"Hidden {i}"
            else:
                layer_name = "Output"
            compression = (1 - len(active_neurons[i]) / full_model.D[i]) * 100
            print(f"    {layer_name:12s}: {full_model.D[i]:4d} → {len(active_neurons[i]):4d} "
                    f"({compression:5.1f}% pruned)")
    
        # Step 2: Extract pruned parameters
   
        m_h_pruned, m_o_pruned, big_b_pruned = full_model.prune_parameters(active_neurons)
        
        # Step 3: Create sparse instance
  
        sparse_model = cls._create_sparse_instance(
            full_model=full_model,
            active_neurons=active_neurons,
            active_weights=active_weights,
            W_stars=W_stars,
            m_h_pruned=m_h_pruned,
            m_o_pruned=m_o_pruned,
            big_b_pruned=big_b_pruned,
        )
        

        return sparse_model
    
    @staticmethod
    def _create_sparse_instance(
        full_model: 'VBNN_improving',
        active_neurons: List[np.ndarray],
        active_weights: List[np.ndarray],
        W_stars: List[np.ndarray],
        m_h_pruned: List[np.ndarray],
        m_o_pruned: np.ndarray,
        big_b_pruned: List[np.ndarray],
    ) -> 'VBNN_SparseTraining':
        """
        Create sparse model instance with pruned parameters.
        
        This is the core initialization that sets up the sparse model.
        Uses object.__new__() to avoid calling __init__ with wrong dimensions.
        """
        # Create empty instance without calling __init__
        sparse_model = object.__new__(VBNN_SparseTraining)
        
        # Mark as sparse model
        sparse_model.is_sparse = True
        
        sparse_model.active_neurons = active_neurons
        sparse_model.active_weights = active_weights
        sparse_model.W_stars = W_stars
        
        # Copy basic attributes and create pruned versions
        sparse_model._copy_from_full_model(full_model, active_neurons)
        
        # Assign pruned parameters
        sparse_model.m_h = m_h_pruned
        sparse_model.m_o = m_o_pruned
        sparse_model.big_b = big_b_pruned

        # Prune parameters related to activation 
        sparse_model._prune_activation_params(full_model, active_neurons)
        # Prune shrinkage and noise parameters
        sparse_model._prune_shrinkage_params(full_model, active_neurons)
        sparse_model._prune_noise_params(full_model, active_neurons)
        
        
        # Initialize tracking
        sparse_model.epoch_no = 0
        sparse_model.elbo_total = []
        
        return sparse_model
    
    def _copy_from_full_model(
        self,
        full_model: 'VBNN_improving',
        active_neurons: List[np.ndarray]  ):
        """
        Copy and transform attributes from full model.
        
        """
        # Store reference to full model architecture (Option 3B)
        self.D_full = full_model.D.copy()
        self.m_h_full = [m.copy() for m in full_model.m_h]
        self.m_o_full = np.copy(full_model.m_o)
        self.big_b_full = [b.copy() for b in full_model.big_b]
        
        # Data - prune input to active features
        self.x_full = full_model.x.copy()
        active_input = active_neurons[0]
        self.x = full_model.x[:, active_input, :]  # Shape: (N, n_active[0], 1)
        self.y = full_model.y.copy()
 
        # Sample handling
        self.N = full_model.N

        # Architecture - pruned dimensions
        self.D = [len(active_neurons[i]) for i in range(len(active_neurons))]
        self.D_in = self.D[0]
        self.D_out = self.D[-1]
        self.L = full_model.L
        

        # Hyperparameters
        self.T = full_model.T
        self.mode = full_model.mode
        self.s_0 = full_model.s_0
        
        # Prior hyperparameters
        self.beta_eta_h_prior = full_model.beta_eta_h_prior
        self.beta_eta_o_prior = full_model.beta_eta_o_prior
        self.alpha_eta_h_prior = full_model.alpha_eta_h_prior
        self.alpha_eta_o_prior = full_model.alpha_eta_o_prior
        self.nu_tau_prior = full_model.nu_tau_prior
        self.nu_psi_prior = full_model.nu_psi_prior
        self.delta_tau_prior = float(full_model.delta_tau_prior)
        self.delta_psi_prior = full_model.delta_psi_prior.copy()


    def _prune_activation_params( self,
        full_model: 'VBNN_improving',
        active_neurons: List[np.ndarray]):
        """
        Prune activation parameters (M, b, S, rho, A) to match active weights.
        """
        self.M = []
        self.b = []
        self.S = []
        self.rho = []
        self.A = []


        for k in range(self.L):
            active_curr = active_neurons[k+1]
            active_prev = active_neurons[k]
         
            # Prune M
            M_k = []
            for m_sample in full_model.M[k]:
                M_k.append(m_sample[np.ix_(active_curr, active_prev)])
            self.M.append(M_k)
            
            # Prune b
            b_k = []
            for b_sample in full_model.b[k]:
                b_k.append(b_sample[active_curr, :])
            self.b.append(b_k)
            
            # # Prune S
            # S_k = []
            # for s_sample in full_model.S[k]:
            #     S_k.append(s_sample[np.ix_(active_curr, active_curr)])
            # self.S.append(np.array(S_k))
            # Prune S - using direct array indexing
            S_k_indices = np.ix_(np.arange(self.N), active_curr, active_curr)
            self.S.append(full_model.S[k][S_k_indices[0], S_k_indices[1], S_k_indices[2]])
      
            # Prune rho
            rho_k = full_model.rho[k][:, active_curr]
            self.rho.append(rho_k)
            
            # Prune A
            A_k = []
            for a_sample in full_model.A[k]:  # a_sample is list of floats
                A_k_sample = [a_sample[i] for i in active_curr]  # Just select active indices!
                A_k.append(A_k_sample)
            self.A.append(A_k)

        # # Test A structure
        # print(f"A[0] type: {type(full_model.A[0])}")  # Should be list
        # print(f"A[0][0] type: {type(full_model.A[0][0])}")  # Should be list
        # print(f"A[0][0][0] type: {type(full_model.A[0][0][0])}")  # Should be float

        # # After pruning
        # print(f"len(A[0][0]): {len(full_model.A[0][0])}")  # Should be D[1]
        # print(f"len(sparse_model.A[0][0]): {len(self.A[0][0])}")  # Should be len(active_neurons[1])
        
        # # Test S structure
        # print(f"S[0] shape: {full_model.S[0].shape}")  # Should be (self.N, D[1], D[1])
        # print(f"S[0] pruned shape: {self.S[0].shape}")

        # # Test M structure
        # print(f"M[0] length: {len(full_model.M[0])}")  # Should be self.N
        # print(f"M[0][0] shape: {full_model.M[0][0].shape}")  # Should be (D[1], D[0])
        # print(f"M[0] pruned length: {len(self.M[0])}")  # Should be self.N
        # print(f"M[0][0] pruned shape: {self.M[0][0].shape}")  # Should be (len(active_neurons[1]), len(active_neurons[0]))      

        # # Test b structure
        # print(f"b[0] length: {len(full_model.b[0])}")  # Should be self.N
        # print(f"b[0][0] shape: {full_model.b[0][0].shape}")  # Should be (D[1], 1)
        # print(f"b[0] pruned length: {len(self.b[0])}")  # Should be self.N
        # print(f"b[0][0] pruned shape: {self.b[0][0].shape}")  # Should be (len(active_neurons[1]), 1)

        # # Test rho structure
        # print(f"rho[0] shape: {full_model.rho[0].shape}")  # Should be (self.N, D[1])
        # print(f"rho[0] pruned shape: {self.rho[0].shape}")  # Should be (self.N, len(active_neurons[1]))


        
        # Cached activations
        self.cached_means = [
            np.zeros((self.N,  len(active_neurons[i]), 1)) for i in range(self.L + 1)
        ]
        self.cached_aats = [
            np.zeros((self.N,  len(active_neurons[i]), len(active_neurons[i]))) 
        for i in range(self.L + 1)
        ]
        self.cache_valid = False

        # self._compute_forward_pass()

        # # print cached means shapes
        # for i in range(len(self.cached_means)):
        #     print(f"cached_means[{i}] shape: {self.cached_means[i].shape}")
        #     print(f"cached_aats[{i}] shape: {self.cached_aats[i].shape}")
   
      

    

    
    def _prune_shrinkage_params(
        self,
        full_model: 'VBNN_improving',
        active_neurons: List[np.ndarray]
    ):
        """
        Prune shrinkage parameters to match active weights.
        
        Global parameters (nu_glob, delta_glob) stay the same (per-layer).
        Local parameters (nu_loc, delta_loc) need pruning (per-weight).
        """
        # Global shrinkage - no pruning needed (one per layer)
        self.nu_glob = full_model.nu_glob.copy()
        self.delta_glob = full_model.delta_glob.copy()
        
        # Local shrinkage - prune to active weights
        self.nu_loc = []
        self.delta_loc = []
        
        for k in range(self.L + 1):
            active_curr = active_neurons[k+1]
            active_prev = active_neurons[k]
            
            # Extract submatrix for active neurons
            # Shape: (n_active[k+1], n_active[k])
            nu_loc_k = full_model.nu_loc[k][np.ix_(active_curr, active_prev)]
            delta_loc_k = full_model.delta_loc[k][np.ix_(active_curr, active_prev)]
            
            self.nu_loc.append(nu_loc_k)
            self.delta_loc.append(delta_loc_k)
    
    def _prune_noise_params(
        self,
        full_model: 'VBNN_improving',
        active_neurons: List[np.ndarray]
    ):
        """
        Prune noise variance parameters.
        
        alpha parameters are scalars (no pruning needed).
        beta parameters need pruning for hidden layers.
        """
        # Alpha parameters - scalars
        self.alpha_h = full_model.alpha_h
        self.alpha_0 = full_model.alpha_0
        
        # Beta for output - no pruning (all outputs kept)
        self.beta_0 = full_model.beta_0.copy()
        
        # Beta for hidden layers - need special handling
        # Original: beta_h shape is (L, D[1]) assuming uniform hidden size
        # After pruning: layers may have different sizes
        
        # Option A: Convert to list of arrays (more flexible)
        self.beta_h = []
        for k in range(self.L):
            active_curr = active_neurons[k+1]
            
            if hasattr(full_model.beta_h, 'shape') and len(full_model.beta_h.shape) == 2:
                # Array format: (L, max_hidden_size)
                beta_h_k = full_model.beta_h[k, active_curr]
            else:
                # List format
                beta_h_k = full_model.beta_h[k][active_curr]
            
            self.beta_h.append(beta_h_k)
        
        # Store full version (Option 3B)
        self.beta_h_full = full_model.beta_h.copy() if hasattr(full_model.beta_h, 'copy') else full_model.beta_h
    
    def _compute_forward_pass(self):
        """
        Compute forward pass through sparse network.
        
        This method uses self.x, self.M, self.b which are all pruned,
        so it works automatically with sparse architecture.
        """
        if self.cache_valid:
            return
        
        
        # Get data based on whether we're using mini-batches
        x_data: np.ndarray

        x_data = self.x
        
        # Input layer
        self.cached_means[0] = x_data  # Shape: (self.N, D[0], 1)
        self.cached_aats[0] = np.array([x @ x.T for x in x_data])
        
        # Hidden layers
        for k in range(self.L):
            means = np.zeros((self.N, self.D[k+1], 1))
            aats = np.zeros((self.N, self.D[k+1], self.D[k+1]))
            
            for n in range(self.N):
                prev_mean = self.cached_means[k][n]
                prev_aat = self.cached_aats[k][n]
                
                # Mean: a[k+1] = b[k] + M[k] @ a[k]
                means[n] = self.b[k][n] + self.M[k][n] @ prev_mean
                
                # Second moment: E[a⊗a]
                aats[n] = (
                    self.S[k][n] + 
                    self.b[k][n] @ self.b[k][n].T +
                    self.M[k][n] @ prev_aat @ self.M[k][n].T +
                    self.M[k][n] @ prev_mean @ self.b[k][n].T +
                    self.b[k][n] @ prev_mean.T @ self.M[k][n].T
                )
            
            self.cached_means[k+1] = means
            self.cached_aats[k+1] = aats
        
        self.cache_valid = True


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

            self.big_b[self.L][j] = solve(np.diag(np.array([1/self.s_0**2] + [inv_mean_IG(self.nu_glob[self.L], self.delta_glob[self.L])*inv_mean_IG(self.nu_loc[self.L][j][i], self.delta_loc[self.L][j][i]) for i in range(self.D[self.L])]))\
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
                    # print(f"Updating layer {k}, sample {n}")
                    # print(f'Before update - M k+1 shape: {self.M[k+1][n].shape}, b k+1 shape: {self.b[k+1][n].shape}, S k+1 shape: {self.S[k+1][n].shape}')
                    # print(f'layer {k+1} D: {self.D[k+1]}, layer {k+2} D: {self.D[k+2]}')
                    # print(f'S k+1 inv shape: {solve(self.S[k+1][n], np.eye(self.D[k+2]), assume_a="pos").shape}')
                    # print(f'M k+1 transpose shape: {self.M[k+1][n].T.shape}, S k+1 inv shape: {solve(self.S[k+1][n], np.eye(self.D[k+2]), assume_a="pos").shape}')
                    term1 = self.M[k+1][n].T@solve(self.S[k+1][n], np.eye(self.D[k+2]), assume_a='pos')@self.M[k+1][n]
                    # print(f'term1 shape: {term1.shape}')
                    term2 = np.sum([(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i]) * self.rho[k+1][n][i] + 1/self.T**2*pg_mean(self.A[k+1][n][i]))*(self.big_b[k+1][i][1:, 1:] + self.m_h[k+1][i][:, 1:].T@self.m_h[k+1][i][:, 1:]) for i in range(self.D[k+2])], axis = 0)
                    # print(f'term2 shape: {term2.shape}')
                    self.S[k][n] =solve(b_h_diag - term1 + term2, np.eye(self.D[k+1]), assume_a = 'pos')
                    # print(f'After S update - S shape: {self.S[k][n].shape}')
                    rho_dot_w = self.m_h[k].squeeze()[:,1:]*self.rho[k][n].reshape(self.D[k+1],1)
                    rho_dot_b = (self.m_h[k].squeeze()[:,0]*self.rho[k][n]).reshape(self.D[k+1],1)
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
        y = np.copy(self.y)
        for j in range(self.N):
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
        
        return elbo_agamma
    
    def elbo(self):
        if self.elbo_total is None:
            raise ValueError("elbo_total must be initialized before calling elbo (via initializing VBNN_algorithm).")
        # self.elbo_total.append((self.elbo_tau() + self.elbo_psi() + self.elbo_eta() + self.elbo_WB() + self.elbo_agamma()))
        self.elbo_total.append((self._elbo_tau() + self._elbo_psi() + self._elbo_eta() + self._elbo_WB() + self._elbo_agamma()).item()) # type: ignore

    
    def cavi_alg(self, epochs: int = 10, EM_step: bool = False):
        """
        Run CAVI algorithm on sparse model.
        
        This is the main training loop for the sparse model.
        
        Args:
            epochs: Number of epochs to train
            EM_step: Whether to update delta_tau_prior (EM step)
        """

    
        # Forward pass
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
        # # Compute ELBO periodically
            if self.epoch_no % 20 == 0 or self.epoch_no == epochs - 1:
                self.elbo()
            self.epoch_no += 1
    


    def _new_delta(self):
        """Update delta_tau_prior (EM step)."""
        
        self.delta_tau_prior = np.sqrt(
            (self.L + 1) * self.nu_tau_prior /
            np.sum([
                self.nu_glob[k] / self.delta_glob[k]**2
                for k in range(self.L + 1)
            ])
        )
    
