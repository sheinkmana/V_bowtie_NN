## Variational Bayesian Bow tie Neural Networks with Shrinkage
This repository provides a Python implementation of the Variational Bow tie neural network (VBNN). 

The VBNN is described and derived in a [Variational Bayesian Bow tie Neural Networks with Shrinkage](https://arxiv.org/abs/2411.11132) paper. 

 ---
 ## Repository has the following files:
 
- VBNN_class.py which contains the main code for the network. 
- helpers.py which contains utilities for easy training and testing.
- test.ipynb notebook which shows how to run VBNN on a simulated example.   

## Overview of the `VBNN` class 

- **Variational Bayes**: Closed-form CAVI updates
- **Sparsity**: Can automatically determine sparse network structures
- **Two initialization modes**: 'laplace' and 'spikeslab' for different sparsity patterns

### Parameters of the class

- **`x`**: Input data (N × D_x numpy array)
- **`y`**: Output data (N × D_y numpy array)
- **`D_a`**: Number of neurons in each hidden layer
- **`L`**: Number of hidden layers
- **`T`**: Temperature parameter (0 < T < 1) controlling sparsity
- **`wb_mode`**: Weight/bias initialization mode ('laplace' or 'spikeslab')
- **`big_S`**: Initial covariance value for stochastic activations

### Internal Components

#### 1. Network Structure
- Input layer: `D[0]` neurons (input dimension)
- Hidden layers: `D[1]` to `D[L]` neurons (each with `D_a` neurons)
- Output layer: `D[L+1]` neurons (output dimension)

#### 2. Variational Parameters

- `m_h[k]`: Mean for hidden layer k weights and biases
- `big_b[k]`: Covariance matrices for weights and biases
- `rho[k]`: Parameter for the Bernulli random variable 
- `M[k]`, `b[k]`: Parameters used to compute mean of stochastic activations
- `S[k]`: Covariance matrices for stochastic activations
- `A[k]`:  Parameters for Pólya-Gamma auxiliary variables
- `m_o`: Mean for output layer weights and biases
- `big_b[L]`: Output layer covariance matrices for weights and biases
- `nu_glob`, `delta_glob`: Global shrinkage parameters 
- `nu_loc`, `delta_loc`: Local shrinkage parameters
- `alpha_h`, `beta_h`: Hidden layer parameters for covariance (Sigma)
- `alpha_0`, `beta_0`: Output layer parameters for covariance (Sigma)

#### 3. Functions

`algorithm(epochs=100, rate=0.000001, EM_step=True)`
Main training loop that alternates between:
1. **Hidden layer updates** (`update_hid_part1`, `update_hid_part2`)
2. **Output layer updates** (`update_out_part1`, `update_out_part2`)
3. **Activation updates** (`update_a`)
4. **Hyperparameter updates** (if `EM_step=True`)
5. **ELBO computation** for convergence monitoring

`predict(x_for_pred, epochs_pred, rate_pred=0.00001)`
- Performs variational inference on new data
- Returns predictive means and uncertainties
- Updates `prediction_mean`, `var_lin` (signal), `var_tot`(total)

`sparse_predict(x_for_pred, epochs_pred, alpha=0.001, rate=0.0000001, epsilon=0.1)`
- Makes predictions with sparse weights only
- More efficient for pruned networks
- Uses `sparse_weighs(alpha=0.0001, epsilon=0.1)` which implements False Discovery Rate (FDR) control for weight selection
