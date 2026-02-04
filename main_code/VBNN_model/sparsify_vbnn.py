import numpy as np
import math
from scipy.stats import norm, invgamma
from scipy.linalg import solve
from .utils import better_sigmoid, pg_mean, inv_mean_IG_eta, smart_log, logcosh
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
    
       
        
    def sparse_weights(self, alpha=1e-2) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Perform sparsification of the VBNN model weights based on FDR thresholding.

        Parameters:
            alpha (float): Desired false discovery rate level.

        Returns:
            active_weights (list of np.ndarray):
                Boolean masks indicating active weights for each layer (including the output layer),
                in the same order as the model's weight matrices.
            active_neurons (list of np.ndarray):
                Indices of active neurons for the input and hidden layers.
                This list has length L+1, where element 0 contains the indices of active input
                features, and elements 1..L contain the indices of active neurons for each of
                the L hidden layers. A neuron is considered "active" if, after thresholding,
                it has at least one nonzero incoming weight and at least one nonzero outgoing
                weight (i.e., it is neither in the zero-input nor zero-output sets).
            W_stars (list of np.ndarray):
                Sparsified weight matrices for each layer, in the same order as `active_weights`.
            active_neurons (list of np.ndarray): Indices of active neurons for each layer.
            W_stars (list of np.ndarray): Sparsified weight matrices for each layer.
        """
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

        active_weights = []
        # Apply sparsity thresholding
        W_stars = []
        for k in range(self.L):
            weights = self.m_h[k].squeeze()[:, 1:]
            stdevs = np.array([np.sqrt(np.diag(self.big_b[k][i][1:, 1:])) for i in range(self.D[k+1])])
            p_vals = np.maximum(1 - norm.cdf(-weights/stdevs), norm.cdf(-weights/stdevs))
            
            # Vectorized active neuron check
            active_neurons = np.array([len(self.rho[k][:, i][(self.rho[k][:, i] < 0.5)]) < 0.99*self.N 
                                        for i in range(self.D[k+1])])
            
            active_weights_layer = (p_vals >= kappa) & active_neurons.reshape(-1,1)
            active_weights.append(active_weights_layer)
            W_star = np.where(active_weights_layer, weights, 0)
            W_stars.append(W_star)

        # Output layer
        # active_neurons_o = np.array([len(self.rho[self.L-1][:, j][(self.rho[self.L-1][:, j] < 0.5)]) < 0.99*self.N 
        #                             for j in range(self.D[self.L])])
        
        active_weights_layer_o = (p_vals_o >= kappa) & active_neurons.reshape(1,-1)
        active_weights.append(active_weights_layer_o)    
        W_star_o = np.where(active_weights_layer_o, weights_o, 0)
        W_stars.append(W_star_o)

        active_neurons = []
        active_neurons.append(np.where(active_weights[0].any(axis=0))[0])
        for k in range(self.L):
            #  W_stars are of the shape D[k+1] x D[k], so to check whether all of the outgoing weights of a neuron are zero, for check fro each of D[k] neurons, whether all of its D[k+1] weights are zero
            zero_outputs = np.all(W_stars[k+1] == 0, axis=0).reshape(-1, 1) # since it is W[k+1] shape should be (D[k+1], 1)
            #  to check whether all of the incoming weights of a neuron are zero, for each of D[k+1] neurons, check whether all of its D[k] weights are zero
            zero_inputs = np.all(W_stars[k] == 0, axis=1).reshape(1, -1) if k+1 < len(W_stars) else np.array([]) # shape should be (1, D[k+1])
            #  if a neuron has zero_inputs, then can set its outgoing weights to zero
            active_weights[k+1] = active_weights[k+1] & ~zero_inputs
            #  if a neuron has zero_outputs, then can set its incoming weights to zero unless it is the output layer
            if k < self.L - 1:
                active_weights[k] = active_weights[k] & ~zero_outputs

            active_neurons.append(np.where(~(zero_outputs | zero_inputs.T))[0])

         
        for k in range(self.L):
            W_stars[k] = np.where(active_weights[k], W_stars[k], 0)

        # add all the output layer neurons as active
        active_neurons.append(np.arange(self.D[self.L + 1]))



        return active_weights, active_neurons, W_stars
    
    

    
    def list_connections(self, W_stars):
        listik = [(f"{k},{j}", f"{k+1},{i}") 
                for k in range(self.L+1) 
                for j in range(self.D[k]) 
                for i in range(self.D[k+1]) 
                if W_stars[k][i, j] != 0]
        # active_weights[k][i, j]
        return listik
    
        
    def prune_parameters(self, active_neurons):
        
        m_h_pruned = []
        big_b_pruned = []

        # Process hidden layers
        for k in range(self.L):
            # Active neurons in current and previous layer
            active_prev = active_neurons[k]      # Shape: (n_active[k],)
            active_curr = active_neurons[k+1]    # Shape: (n_active[k+1],)
            
            n_active_prev = len(active_prev)
            n_active_curr = len(active_curr)
            
            # Extract m_h[k] for active neurons
            # Original m_h[k]: (D[k+1], 1, D[k]+1) where [:, 0, 0] is bias, [:, 0, 1:] are weights
            m_h_k_pruned = np.zeros((n_active_curr, 1, n_active_prev + 1))
            
            # Extract biases for active current neurons
            m_h_k_pruned[:, 0, 0] = self.m_h[k][active_curr, 0, 0]
            
            # Extract weights for active neuron pairs
            # Need to select: rows=active_curr, middle dim=0, columns=active_prev+1 (shift by 1 for bias)
            weight_indices = active_prev + 1
            for i, curr_idx in enumerate(active_curr):
                m_h_k_pruned[i, 0, 1:] = self.m_h[k][curr_idx, 0, weight_indices]
            
            m_h_pruned.append(m_h_k_pruned)
            
            # Extract big_b[k] for active neurons
            # Original big_b[k]: (D[k+1], D[k]+1, D[k]+1)
            # For each neuron i in D[k+1], big_b[k][i] is a (D[k]+1, D[k]+1) covariance matrix
            big_b_k_pruned = np.zeros((n_active_curr, n_active_prev + 1, n_active_prev + 1))
            
            for i, neuron_idx in enumerate(active_curr):
                # Extract the covariance block for this active neuron
                # Indices: [0] for bias, [active_prev + 1] for active weights
                cov_indices = np.concatenate([[0], active_prev + 1])
                big_b_k_pruned[i] = self.big_b[k][neuron_idx][np.ix_(cov_indices, cov_indices)]
            
            big_b_pruned.append(big_b_k_pruned)

        # Process output layer
        active_last_hidden = active_neurons[self.L]  # Shape: (n_active[L],)
        n_active_last = len(active_last_hidden)
        n_outputs = self.D[self.L + 1]

        # Extract m_o
        # Original m_o: (D[L+1], 1, D[L]+1) 
        m_o_pruned = np.zeros((n_outputs, 1, n_active_last + 1))

        # Bias
        m_o_pruned[:, 0, 0] = self.m_o[:, 0, 0]

        # Weights for active neurons
        weight_indices_o = active_last_hidden + 1
        for i in range(n_outputs):
            m_o_pruned[i, 0, 1:] = self.m_o[i, 0, weight_indices_o]

        # Extract big_b[L] (output layer covariances)
        # Original big_b[L]: (D[L+1], D[L]+1, D[L]+1)
        big_b_L_pruned = np.zeros((n_outputs, n_active_last + 1, n_active_last + 1))

        for i in range(n_outputs):
            cov_indices_o = np.concatenate([[0], active_last_hidden + 1])
            big_b_L_pruned[i] = self.big_b[self.L][i][np.ix_(cov_indices_o, cov_indices_o)]

        big_b_pruned.append(big_b_L_pruned)




        # Print compression statistics
        print("\nPruned Model Dimensions:")
        print(f"Input features: {self.D[0]} -> {len(active_neurons[0])}")
        for k in range(self.L):
            print(f"Hidden layer {k+1}: {self.D[k+1]} -> {len(active_neurons[k+1])}")
        print(f"Output layer: {self.D[self.L+1]} (unchanged)")

        # print("\nPruned Parameter Shapes:")
        # for k in range(self.L):
        #     print(f"m_h[{k}]: {m_h_pruned[k].shape}")
        #     print(f"big_b[{k}]: {big_b_pruned[k].shape}")
        # print(f"m_o: {m_o_pruned.shape}")
        # print(f"big_b[L]: {big_b_pruned[-1].shape}")

        # print('previous shapes:')
        # for k in range(self.L):
        #     print(f"m_h[{k}]: {self.m_h[k].shape}")
        #     print(f"big_b[{k}]: {self.big_b[k].shape}")
        # print(f"m_o: {self.m_o.shape}")
        # print(f"big_b[L]: {self.big_b[-1].shape}")


        return m_h_pruned, m_o_pruned, big_b_pruned



    def mask_parameters(self, alpha = 1e-2):
        """
        Zero out inactive weights based on FDR sparsification.
        Model keeps full dimensions but irrelevant weights = 0.
        """
        _, active_neurons, _ = self.sparse_weights(alpha)
 
        # Zero out m_h (hidden weights)
        for k in range(self.L):
            active_curr = active_neurons[k+1]
            active_prev = active_neurons[k]
            
            # Weight mask
            mask_w = np.zeros((self.D[k+1], 1, self.D[k]+1))
            mask_w[active_curr, 0, 0] = 1
            for i in active_curr:
                for j in active_prev:
                    mask_w[i, 0, j+1] = 1
            
            self.m_h[k] *= mask_w
            
            
            
        # Zero out m_o (output weights)
        mask_o = np.zeros((self.D[self.L+1], 1, self.D[self.L]+1))
        mask_o[:, 0, 0] = 1  # All output biases
        for j in active_neurons[self.L]:
            mask_o[:, 0, j+1] = 1  # Active connections
        
        self.m_o *= mask_o

        total_params = sum(self.D[i] * self.D[i+1] for i in range(self.L + 1))
        active_params = sum(len(active_neurons[i]) * len(active_neurons[i+1]) 
                       for i in range(self.L + 1))
        print(f"\nTotal parameters: {total_params}, Active parameters after masking: {active_params}, Compression ratio: {total_params/active_params:.2f}")
    

        
        
    def _compute_forward_pass_predict_pruned(self, x_new):
        """
        Compute forward pass for prediction using pruned model dimensions.
        
        Parameters:
            x_new: Input data of shape (N_pred, len(active_neurons[0]), 1) - already filtered to active features
            active_neurons: List of active neuron indices from sparse_weights()
        """
        if self.cache_valid_predict:
            return
        
        N_pred = x_new.shape[0]
        
        # Ensure b_predict and M_predict are initialized
        if self.b_predict is None or self.M_predict is None or self.S_predict is None:
            raise ValueError("b_predict and M_predict must be initialized before calling _compute_forward_pass_predict.")
        
        # Layer 0 (input) - active input features only
        self.cached_means_predict[0] = x_new  # Shape: (N_pred, len(active_neurons[0]), 1)
        self.cached_aats_predict[0] = np.array([x @ x.T for x in x_new])  # Shape: (N_pred, len(active_neurons[0]), len(active_neurons[0]))

        # Forward pass through hidden layers
        for layer in range(1, self.L + 1):
            n_active_curr = len(self.active_neurons[layer])  # Number of active neurons in current layer
            
            means = np.zeros((N_pred, n_active_curr, 1))
            aats = np.zeros((N_pred, n_active_curr, n_active_curr))
            
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

    
    def sparse_initforpredict_pruned(self, x_for_pred, alpha = 0.001):

        if self.A_predict is None:
            raise ValueError("A_predict must be initialized before calling prediction (via initializing VBNN_algorithm).")
     
       
        _, self.active_neurons, _ = self.sparse_weights(alpha)
        N_pred = x_for_pred.shape[0]
        active_input_features = self.active_neurons[0]
        x_new = np.copy(x_for_pred[:, active_input_features]).reshape(N_pred, len(active_input_features), 1)
        self.m_h_pruned, self.m_o_pruned, self.big_b_pruned= self.prune_parameters(self.active_neurons)
        self.M_predict = []
        self.b_predict = []
        self.S_predict = []
        self.rho_predict = []
        x_init = np.copy(x_new)

        # Process each hidden layer with pruned dimensions
        for k in range(self.L):
            n_active_curr = len(self.active_neurons[k+1])  # Active neurons in current layer
            
            # Initialize rho_predict: shape (N_pred, n_active_curr)
            self.rho_predict.append(
                np.array([[better_sigmoid((self.m_h_pruned[k][i][0, 0] + 
                                        self.m_h_pruned[k][i][0, 1:]@x_init[n]).item()/self.T) 
                        for i in range(n_active_curr)] 
                        for n in range(N_pred)])
            )
            
            # Initialize M_predict: list of N_pred arrays, each of shape (n_active_curr, n_active_prev)
            self.M_predict.append(
                [self.m_h_pruned[k].squeeze()[:, 1:] * self.rho_predict[k][j].reshape(n_active_curr, 1) 
                for j in range(N_pred)]
            )
            
            # Initialize b_predict: list of N_pred arrays, each of shape (n_active_curr, 1)
            self.b_predict.append(
                [(self.m_h_pruned[k].squeeze()[:, 0] * self.rho_predict[k][j]).reshape(n_active_curr, 1) 
                for j in range(N_pred)]
            )
            
            # Forward pass to next layer
            x_init = np.array([np.copy(self.M_predict[k][n])@np.copy(x_init[n]) + 
                            np.copy(self.b_predict[k][n]) 
                            for n in range(N_pred)]).reshape(N_pred, n_active_curr, 1)
            
            # Initialize S_predict: array of shape (N_pred, n_active_curr, n_active_curr)
            self.S_predict.append(np.array([np.mean(self.S).item()*np.eye(n_active_curr)]*N_pred))



            # Update cached prediction arrays with pruned dimensions
        self.cached_means_predict = [
        np.zeros((N_pred, len(self.active_neurons[i]), 1)) for i in range(self.L + 1)
        ]

        self.cached_aats_predict = [
        np.zeros((N_pred, len(self.active_neurons[i]), len(self.active_neurons[i]))) 
        for i in range(self.L + 1)
        ]
   
        self.cache_valid_predict = False
        
        self._compute_forward_pass_predict_pruned(x_new)

        self.A_predict = []
        
        for k in range(self.L):
            n_active_curr = len(self.active_neurons[k+1])  # Active neurons in current layer
            # n_active_prev = len(self.active_neurons[k])    # Active neurons in previous layer
            
            self.A_predict.append([])
            
            for j in range(N_pred):
                # Get cached activations for sample j in layer k
                # Shape: (n_active_prev, 1)
                tobm = self.cached_means_predict[k][j]
                
                # Build augmented matrix with bias term
                # big_matrix shape: (n_active_prev+1, n_active_prev+1)
                # Top-left: scalar 1
                # Top-right: (1, n_active_prev) = tobm.T
                # Bottom-left: (n_active_prev, 1) = tobm
                # Bottom-right: (n_active_prev, n_active_prev) = aa^T
                big_matrix = np.hstack((
                    np.vstack((1, tobm)), 
                    np.vstack((tobm.T, np.copy(self.cached_aats_predict[k][j])))
                ))
                
                # Compute A for each active neuron in current layer
                # self.m_h_pruned[k][i]: shape (1, n_active_prev+1)
                # self.big_b_pruned[k][i]: shape (n_active_prev+1, n_active_prev+1)
                A_values = [
                    1/self.T * math.sqrt(
                        ((self.big_b_pruned[k][i] + self.m_h_pruned[k][i].T @ self.m_h_pruned[k][i]) * big_matrix).sum()
                    ) 
                    for i in range(n_active_curr)
                ]
                
                self.A_predict[k].append(A_values)
        
        # Convert to arrays for consistency
        for k in range(self.L):
            self.A_predict[k] = np.array(self.A_predict[k])  # Shape: (N_pred, n_active[k+1])

        print("\n init done:")
        for layer in range(self.L + 1):
            print(f"Layer {layer}: cached_means shape = {self.cached_means_predict[layer].shape}, "
                f"expected = (N_pred, {len(self.active_neurons[layer])}, 1)")
            print(f"Layer {layer}: cached_aats shape = {self.cached_aats_predict[layer].shape}, "
                f"expected = (N_pred, {len(self.active_neurons[layer])}, {len(self.active_neurons[layer])})")
            
        for layer in range(self.L):
            print(f"Layer {layer}: A_predict shape = {self.A_predict[layer].shape}, "
                f"expected = (N_pred, {len(self.active_neurons[layer+1])})")
            print(f"Layer {layer}: M_predict length = {len(self.M_predict[layer])}, "
                f"each shape = {self.M_predict[layer][0].shape}, expected = ({len(self.active_neurons[layer+1])}, {len(self.active_neurons[layer])})")
            print(f"Layer {layer}: b_predict length = {len(self.b_predict[layer])}, "
                f"each shape = {self.b_predict[layer][0].shape}, expected = ({len(self.active_neurons[layer+1])}, 1)")
            print(f"Layer {layer}: S_predict shape = {self.S_predict[layer].shape}, "
                f"expected = (N_pred, {len(self.active_neurons[layer+1])}, {len(self.active_neurons[layer+1])})")
        self.elbo_pred = []

    def sparse_predict_pruned(self, x_for_pred, epochs_pred, alpha = 0.001, rate_pred = 0.00001):

        N_pred = x_for_pred.shape[0]
    
    
        self.sparse_initforpredict_pruned(x_for_pred, alpha)
   # Filter input to active features
        active_input_features = self.active_neurons[0]
        x_new = np.copy(x_for_pred[:, active_input_features]).reshape(
            N_pred, len(active_input_features), 1
        )

        for ep_pred in range(epochs_pred):
            for k in reversed(range(self.L)):
                n_active_curr = len(self.active_neurons[k+1])
                n_active_next = len(self.active_neurons[k+2]) if k < self.L - 1 else None
                
                if k == self.L - 1:
                    # Last hidden layer: no layer above to condition on
                    b_h_diag_predict = np.diag([1/inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) for i in range(n_active_curr)])
                    
                    for j in range(N_pred):
                        self.S_predict[k][j] = np.copy(b_h_diag_predict)
                        # Update M and b based on current rho
                        rho_dot_w = (self.m_h_pruned[k].squeeze()[:, 1:] * self.rho_predict[k][j].reshape(n_active_curr, 1))
                        rho_dot_b = ((self.m_h_pruned[k].squeeze()[:, 0] * self.rho_predict[k][j]).reshape(n_active_curr, 1))
                        self.b_predict[k][j] = np.copy(rho_dot_b)
                        self.M_predict[k][j] = np.copy(rho_dot_w)
                
                else:
                    # Interior layers
                    b_h_diag_predict = np.diag([
                        inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])  
                        for i in range(n_active_curr)
                    ])
                    for j in range(N_pred):
                        term1 = (self.M_predict[k+1][j].T @ 
                                    solve(self.S_predict[k+1][j], 
                                        np.eye((n_active_next if n_active_next is not None else 1)),  # Fixed: k+2 not k+1
                                        assume_a='pos') @ 
                                    self.M_predict[k+1][j])
                            
                        # Term 2: sum over neurons in layer k+2
                        term2 = np.sum([
                            (inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i]) * 
                            self.rho_predict[k+1][j][i] + 
                            1/self.T**2 * pg_mean(self.A_predict[k+1][j][i])) * 
                            (self.big_b_pruned[k+1][i][1:, 1:] + 
                            self.m_h_pruned[k+1][i][0, 1:].T @ self.m_h_pruned[k+1][i][0, 1:])
                            for i in range(n_active_next if n_active_next is not None else 0)], axis=0)
                        # Add diagnostic

                        
                    # DIAGNOSTIC
                        if ep_pred == 0 and k == 0 and j == 0:
                            print("\n=== DIAGNOSTIC at ep=0, k=0, j=0 ===")
                            for i in range(min(3, n_active_next if n_active_next is not None else 1)):
                                matrix_part = self.big_b_pruned[k+1][i][1:, 1:]
                                print(f"big_b_pruned[{k+1}][{i}][1:,1:] eigenvalues: {np.linalg.eigvalsh(matrix_part)[:5]}")
                                print(f"big_b_pruned[{k+1}][{i}][1:,1:] trace: {np.trace(matrix_part)}")
                            print(f"b_h_diag_predict:\n{np.diag(b_h_diag_predict)[:5]}")
                            print(f"term1 diagonal:\n{np.diag(term1)[:5]}")
                            print(f"term2 diagonal:\n{np.diag(term2)[:5]}")
                            precision = b_h_diag_predict - term1 + term2
                            print(f"precision diagonal:\n{np.diag(precision)[:5]}")
                            print(f"precision eigenvalues (smallest 5):\n{np.sort(np.linalg.eigvalsh(precision))[:5]}")
                            print(f"M_predict[1][0] norm: {np.linalg.norm(self.M_predict[k+1][j])}")
                            print(f"S_predict[1][0] condition: {np.linalg.cond(self.S_predict[k+1][j])}")
                    
                                                # Before solve operation
                  
                    
                        self.S_predict[k][j] = solve( b_h_diag_predict - term1 + term2, np.eye(n_active_curr), assume_a='pos')
                            
                    
                        # self.S_predict[k][j] = solve( b_h_diag_predict - term1 + term2, np.eye(n_active_curr), assume_a='pos')
                        # print('shape of S_predict after update:', self.S_predict[k][j].shape)
                        # Update rho-weighted weights and biases
                        rho_dot_w = (self.m_h_pruned[k].squeeze()[:, 1:] * self.rho_predict[k][j].reshape(n_active_curr, 1))
                        rho_dot_b = ((self.m_h_pruned[k].squeeze()[:, 0] * self.rho_predict[k][j]).reshape(n_active_curr, 1))
                        
                        # Compute sum for b update
                        sum_for_b = 0
                        if n_active_next is not None:
                            sum_for_b = np.sum([-(inv_mean_IG_eta(self.alpha_h, self.beta_h[k+1][i]) * self.rho_predict[k+1][j][i] + 
                                1/self.T**2 * pg_mean(self.A_predict[k+1][j][i])) *  (self.big_b_pruned[k+1][i][1:, :1] +  self.m_h_pruned[k+1][i][0, 0] * self.m_h_pruned[k+1][i][0, 1:].reshape(self.big_b_pruned[k+1][i][1:, :1].shape)) + 
                                1/self.T * (self.rho_predict[k+1][j][i] - 0.5) * self.m_h_pruned[k+1][i][0, 1:].reshape(self.big_b_pruned[k+1][i][1:, :1].shape) for i in range(n_active_next)], axis=0)
               


                        self.b_predict[k][j] = self.S_predict[k][j] @ (
                            b_h_diag_predict @ rho_dot_b + 
                            self.M_predict[k+1][j].T @ 
                            solve(
                                self.S_predict[k+1][j],
                                self.b_predict[k+1][j],  # Direct solve, no identity matrix
                                assume_a='pos'
                            ) +
                            sum_for_b
                        )

                        self.M_predict[k][j] = self.S_predict[k][j] @ b_h_diag_predict @ rho_dot_w

            self.cache_valid_predict = False
            self._compute_forward_pass_predict_pruned(x_new)

            for k in range(self.L):
                n_active_curr = len(self.active_neurons[k+1])
                n_active_prev = len(self.active_neurons[k])
                
                self.A_predict[k] = []
                
                for j in range(N_pred):
                    # Build augmented matrix for A computation
                    tobm = self.cached_means_predict[k][j]
                    big_matrix = np.hstack((np.vstack((1, tobm)), np.vstack((tobm.T, np.copy(self.cached_aats_predict[k][j])))))
                    
                    # Update A[k][j]
                    A_values = [1/self.T * math.sqrt(((self.big_b_pruned[k][i] + self.m_h_pruned[k][i].T @ self.m_h_pruned[k][i]) * big_matrix).sum() ) for i in range(n_active_curr)]
                    self.A_predict[k].append(A_values)
                    
                    # Update rho[k][j]
                    rho_pr_slot = []
                    a_mean_prev = np.vstack((1, self.cached_means_predict[k][j]))
                    aat_mean_prev = np.hstack(( a_mean_prev,  np.vstack((a_mean_prev[1:].T, self.cached_aats_predict[k][j]))))
                    
                    for i in range(n_active_curr):
                        aat_shifted = aat_mean_prev @ np.hstack(( self.b_predict[k][j][i], self.M_predict[k][j][i])).reshape(n_active_prev + 1, 1)
                        
                        eta = inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i])
                        
                        rho_update = better_sigmoid(( -0.5 * eta * ((self.big_b_pruned[k][i] +self.m_h_pruned[k][i].T @ self.m_h_pruned[k][i]) *(aat_mean_prev).T).sum() +  eta * self.m_h_pruned[k][i] @ aat_shifted +  1/self.T * self.m_h_pruned[k][i] @ a_mean_prev).item())
                        
                        rho_pr_slot.append(rho_update)
                    
                    self.rho_predict[k][j] = np.copy(rho_pr_slot)
                
                # Convert A to array
                self.A_predict[k] = np.array(self.A_predict[k])


                # ===============================================================
            # ELBO COMPUTATION (every 10 iterations)
            # ===============================================================
            if ep_pred % 10 == 0:
                temp_sum = 0
                
                for j in range(N_pred):
                    for k in range(self.L):
                        n_active_curr = len(self.active_neurons[k+1])
                        n_active_prev = len(self.active_neurons[k])
                        
                        a_mean_prev = self.cached_means_predict[k][j]
                        aat_prev = self.cached_aats_predict[k][j]
                        aat_current = self.cached_aats_predict[k+1][j]
                        a_current = self.cached_means_predict[k+1][j]
                        
                        for i in range(n_active_curr):
                            aat_shifted = (self.b_predict[k][j][i] * a_mean_prev + 
                                        (aat_prev @ self.M_predict[k][j][i]).reshape(n_active_prev, 1))
                            
                            big_trace = (self.big_b_pruned[k][i][0, 0] + 
                                        self.m_h_pruned[k][i][0, 0]**2 + 
                                        2 * (self.big_b_pruned[k][i][0, 1:] + 
                                            self.m_h_pruned[k][i][0, 0] * self.m_h_pruned[k][i][0, 1:]) @ a_mean_prev + 
                                        ((self.big_b_pruned[k][i][1:, 1:] + 
                                        self.m_h_pruned[k][i][0, 1:].T @ self.m_h_pruned[k][i][0, 1:]) * 
                                        (aat_prev).T).sum())
                            
                            small_trace = (self.m_h_pruned[k][i][0, 0]**2 + 
                                        2 * self.m_h_pruned[k][i][0, 0] * self.m_h_pruned[k][i][0, 1:] @ a_mean_prev + 
                                        ((self.m_h_pruned[k][i][0, 1:].T @ self.m_h_pruned[k][i][0, 1:]) * 
                                        (a_mean_prev @ a_mean_prev.T).T).sum())
                            
                            temp_sum += (-0.5 * inv_mean_IG_eta(self.alpha_h, self.beta_h[k][i]) * 
                                        ((self.rho_predict[k][j][i] * 
                                        (self.m_h_pruned[k][i][0, 1:] @ a_mean_prev + 
                                        self.m_h_pruned[k][i][0, 0]) - a_current[i])**2 + 
                                        aat_current[i, i] - a_current[i]**2 + 
                                        self.rho_predict[k][j][i] * big_trace - 
                                        self.rho_predict[k][j][i]**2 * small_trace + 
                                        self.rho_predict[k][j][i] * self.m_h_pruned[k][i][0, 1:] @ 
                                        (a_current[i] * a_mean_prev - aat_shifted)) + 
                                        1/self.T * (self.rho_predict[k][j][i] - 0.5) * 
                                        (self.m_h_pruned[k][i][0, 1:] @ a_mean_prev + 
                                        self.m_h_pruned[k][i][0, 0]) - 
                                        1/(2 * self.T**2) * pg_mean(self.A_predict[k][j][i]) * big_trace - 
                                        smart_log(self.rho_predict[k][j][i]) - 
                                        smart_log((1 - self.rho_predict[k][j][i])) - 
                                        logcosh(self.A_predict[k][j][i] * 0.5) + 
                                        0.5 * pg_mean(self.A_predict[k][j][i]) * self.A_predict[k][j][i]**2)
                        
                        temp_sum += 0.5 * np.linalg.slogdet(self.S_predict[k][j])[1]
                
                self.elbo_pred.append(temp_sum)
                print(f"Epoch {ep_pred}: ELBO = {temp_sum}")
            
            # ===============================================================
            # CONVERGENCE CHECK
            # ===============================================================
            if len(self.elbo_pred) > 3:
                if np.abs(1 - self.elbo_pred[-1] / self.elbo_pred[-2]) < rate_pred:
                    print(f"Converged at epoch {ep_pred}")
                    break
    
                # Prediction mean
        self.prediction_mean = np.array([[(self.m_o_pruned[j][0, 1:] @ self.cached_means_predict[self.L][i] + self.m_o_pruned[j][0, 0]).item() for j in range(self.D[self.L + 1])]for i in range(N_pred)])
        
        # Collect final hidden activations
        pred_mean = []
        for i in range(N_pred):
            pred_mean.append(self.cached_means_predict[self.L][i])
        
        # Linear variance
        self.var_lin = np.array([[self.big_b_pruned[-1][j][0, 0] + 2 * self.big_b_pruned[-1][j][0, 1:].reshape(1, len(self.active_neurons[self.L])) @ pred_mean[i] + 
            ((self.big_b_pruned[-1][j][1:, 1:] +  self.m_o_pruned[j][0, 1:].T @ self.m_o_pruned[j][0, 1:]) *  self.cached_aats_predict[self.L][i].T).sum() -  self.m_o_pruned[j][0, 1:] @ (pred_mean[i] @ pred_mean[i].T) @ self.m_o_pruned[j][0, 1:].T
            for i in range(N_pred)] for j in range(self.D[self.L + 1]) ]).reshape(self.prediction_mean.shape)
        
        # Total variance (linear + noise)
        self.var_tot = np.array([ self.var_lin[:, i] + invgamma.mean(a=self.alpha_0, loc=0, scale=self.beta_0[i]) for i in range(self.D[-1])]).reshape(self.prediction_mean.shape)
        

        print(f"Prediction mean shape: {self.prediction_mean.shape}")
        print(f"Variance shape: {self.var_tot.shape}")
        
 
