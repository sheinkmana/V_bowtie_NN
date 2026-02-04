# VBNN Model Parameters Documentation

## Table of Contents
1. [Architecture Parameters](#architecture-parameters)
2. [Weight and Bias Parameters](#weight-and-bias-parameters)
3. [Activation Parameters](#activation-parameters)
4. [Shrinkage Parameters](#shrinkage-parameters)
5. [Noise Variance Parameters](#noise-variance-parameters)
6. [Cached Variables](#cached-variables)
7. [Prediction Parameters](#prediction-parameters)
8. [Pruned Model Parameters](#pruned-model-parameters)

---

## Architecture Parameters

### Full Model
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.D` | `list[int]` | Length `L+2` | Layer dimensions: `[D_in, D_H, D_H, ..., D_H, D_out]` |
| `self.L` | `int` | Scalar | Number of hidden layers |
| `self.N` | `int` | Scalar | Number of training samples |
| `self.D_in` | `int` | Scalar | Input dimension = `D[0]` |
| `self.D_out` | `int` | Scalar | Output dimension = `D[L+1]` |

### Pruned Model
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `active_neurons` | `list[np.ndarray]` | Length `L+2` | Active neuron indices for each layer |
| `active_neurons[i]` | `np.ndarray` | Shape `(n_active[i],)` | Indices of active neurons in layer `i` |
| `n_active[i]` | `int` | Variable | Number of active neurons in layer `i` = `len(active_neurons[i])` |

---

## Weight and Bias Parameters (Variational Posteriors)

### Hidden Layers (Full Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.m_h` | `list[np.ndarray]` | Length `L` | Mean parameters for hidden layers |
| `self.m_h[k]` | `np.ndarray` | Shape `(D[k+1], 1, D[k]+1)` | Mean for layer `k`: `[:, 0, 0]` = bias, `[:, 0, 1:]` = weights |
| `self.big_b` | `list[np.ndarray]` | Length `L+1` | Covariance matrices for all layers |
| `self.big_b[k]` | `np.ndarray` | Shape `(D[k+1], D[k]+1, D[k]+1)` | Covariance for layer `k`, one matrix per neuron |
| `self.big_b[k][i]` | `np.ndarray` | Shape `(D[k]+1, D[k]+1)` | Covariance for neuron `i` in layer `k+1` |

### Output Layer (Full Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.m_o` | `np.ndarray` | Shape `(D[L+1], 1, D[L]+1)` | Mean for output layer: `[:, 0, 0]` = bias, `[:, 0, 1:]` = weights |
| `self.big_b[L]` | `np.ndarray` | Shape `(D[L+1], D[L]+1, D[L]+1)` | Covariance for output layer |

### Hidden Layers (Pruned Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `m_h_pruned` | `list[np.ndarray]` | Length `L` | Pruned mean parameters for hidden layers |
| `m_h_pruned[k]` | `np.ndarray` | Shape `(n_active[k+1], 1, n_active[k]+1)` | Pruned mean for layer `k` |
| `big_b_pruned` | `list[np.ndarray]` | Length `L+1` | Pruned covariance matrices |
| `big_b_pruned[k]` | `np.ndarray` | Shape `(n_active[k+1], n_active[k]+1, n_active[k]+1)` | Pruned covariance for layer `k` |
| `big_b_pruned[k][i]` | `np.ndarray` | Shape `(n_active[k]+1, n_active[k]+1)` | Covariance for active neuron `i` |

### Output Layer (Pruned Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `m_o_pruned` | `np.ndarray` | Shape `(D[L+1], 1, n_active[L]+1)` | Pruned mean for output (all output neurons kept) |
| `big_b_pruned[L]` | `np.ndarray` | Shape `(D[L+1], n_active[L]+1, n_active[L]+1)` | Pruned covariance for output layer |

---

## Activation Parameters (for Hidden Layers)

### Training (Full Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.rho` | `list[np.ndarray]` | Length `L` | Binary activation probabilities |
| `self.rho[k]` | `np.ndarray` | Shape `(N, D[k+1])` or `(sample_size, D[k+1])` | Activation probabilities for layer `k` |
| `self.A` | `list[np.ndarray]` | Length `L` | Pólya-Gamma auxiliary variables |
| `self.A[k]` | `np.ndarray` | Shape `(N, D[k+1])` or `(sample_size, D[k+1])` | Auxiliary variables for layer `k` |
| `self.b` | `list[list[np.ndarray]]` | Length `L` | Bias terms for activations |
| `self.b[k]` | `list[np.ndarray]` | Length `N` (or `sample_size`) | List of bias vectors |
| `self.b[k][n]` | `np.ndarray` | Shape `(D[k+1], 1)` | Bias for sample `n` in layer `k` |
| `self.M` | `list[list[np.ndarray]]` | Length `L` | Weight matrices for activations |
| `self.M[k]` | `list[np.ndarray]` | Length `N` (or `sample_size`) | List of weight matrices |
| `self.M[k][n]` | `np.ndarray` | Shape `(D[k+1], D[k])` | Weights for sample `n` in layer `k` |
| `self.S` | `list[np.ndarray]` | Length `L` | Covariance matrices for activations |
| `self.S[k]` | `np.ndarray` | Shape `(N, D[k+1], D[k+1])` or `(sample_size, D[k+1], D[k+1])` | Covariances for layer `k` |

### Prediction (Full Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.rho_predict` | `list[np.ndarray]` | Length `L` | Activation probabilities for prediction |
| `self.rho_predict[k]` | `np.ndarray` | Shape `(N_pred, D[k+1])` | Activation probabilities for layer `k` |
| `self.A_predict` | `list[np.ndarray]` | Length `L` | Pólya-Gamma variables for prediction |
| `self.A_predict[k]` | `np.ndarray` | Shape `(N_pred, D[k+1])` | Auxiliary variables for layer `k` |
| `self.b_predict` | `list[list[np.ndarray]]` | Length `L` | Bias terms for prediction |
| `self.b_predict[k]` | `list[np.ndarray]` | Length `N_pred` | List of bias vectors |
| `self.b_predict[k][n]` | `np.ndarray` | Shape `(D[k+1], 1)` | Bias for prediction sample `n` |
| `self.M_predict` | `list[list[np.ndarray]]` | Length `L` | Weight matrices for prediction |
| `self.M_predict[k]` | `list[np.ndarray]` | Length `N_pred` | List of weight matrices |
| `self.M_predict[k][n]` | `np.ndarray` | Shape `(D[k+1], D[k])` | Weights for prediction sample `n` |
| `self.S_predict` | `list[np.ndarray]` | Length `L` | Covariance matrices for prediction |
| `self.S_predict[k]` | `np.ndarray` | Shape `(N_pred, D[k+1], D[k+1])` | Covariances for layer `k` |

### Prediction (Pruned Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.rho_predict` | `list[np.ndarray]` | Length `L` | Pruned activation probabilities |
| `self.rho_predict[k]` | `np.ndarray` | Shape `(N_pred, n_active[k+1])` | Activation probabilities for active neurons |
| `self.b_predict[k]` | `list[np.ndarray]` | Length `N_pred` | Pruned bias terms |
| `self.b_predict[k][n]` | `np.ndarray` | Shape `(n_active[k+1], 1)` | Bias for active neurons in sample `n` |
| `self.M_predict[k]` | `list[np.ndarray]` | Length `N_pred` | Pruned weight matrices |
| `self.M_predict[k][n]` | `np.ndarray` | Shape `(n_active[k+1], n_active[k])` | Weights between active neurons |
| `self.S_predict[k]` | `np.ndarray` | Shape `(N_pred, n_active[k+1], n_active[k+1])` | Pruned covariances |

---

## Shrinkage Parameters

### Global Shrinkage (Full Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.nu_glob` | `np.ndarray` | Shape `(L+1,)` | Global shrinkage degrees of freedom (one per layer) |
| `self.delta_glob` | `np.ndarray` | Shape `(L+1,)` | Global shrinkage scales (one per layer) |

### Local Shrinkage (Full Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.nu_loc` | `list[np.ndarray]` | Length `L+1` | Local shrinkage degrees of freedom |
| `self.nu_loc[k]` | `np.ndarray` | Shape `(D[k+1], D[k])` | DOF for each weight in layer `k` |
| `self.delta_loc` | `list[np.ndarray]` | Length `L+1` | Local shrinkage scales |
| `self.delta_loc[k]` | `np.ndarray` | Shape `(D[k+1], D[k])` | Scale for each weight in layer `k` |

### Prior Hyperparameters (Fixed)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.delta_tau_prior` | `float` | Scalar | Global shrinkage prior scale |
| `self.delta_psi_prior` | `np.ndarray` | Shape `(L+2,)` | Local shrinkage prior scales per layer |
| `self.s_0` | `float` | Scalar | Prior scale for bias terms |
| `self.nu_tau_prior` | `float` | Scalar | Global shrinkage prior DOF (fixed at -1.5) |
| `self.nu_psi_prior` | `float` | Scalar | Local shrinkage prior DOF (fixed at -1.5) |

---

## Noise Variance Parameters

### Hidden Layers (Full Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.alpha_h` | `float` | Scalar | Shape parameter for hidden noise (shared across layers) |
| `self.beta_h` | `np.ndarray` | Shape `(L, D[1])` | Scale parameters for hidden layer noise |
| `self.alpha_eta_h_prior` | `float` | Scalar | Prior shape (fixed at 2.0) |
| `self.beta_eta_h_prior` | `float` | Scalar | Prior scale for hidden noise |

### Output Layer (Full Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.alpha_0` | `float` | Scalar | Shape parameter for output noise |
| `self.beta_0` | `np.ndarray` | Shape `(D[L+1],)` | Scale parameters for output noise |
| `self.alpha_eta_o_prior` | `float` | Scalar | Prior shape (fixed at 2.0) |
| `self.beta_eta_o_prior` | `float` | Scalar | Prior scale for output noise |

---

## Cached Variables (for Efficient Computation)

### Training (Full Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.cached_means` | `list[np.ndarray]` | Length `L+1` | Cached mean activations |
| `self.cached_means[i]` | `np.ndarray` | Shape `(N, D[i], 1)` or `(sample_size, D[i], 1)` | Mean activations for layer `i` |
| `self.cached_aats` | `list[np.ndarray]` | Length `L+1` | Cached second moment matrices |
| `self.cached_aats[i]` | `np.ndarray` | Shape `(N, D[i], D[i])` or `(sample_size, D[i], D[i])` | Second moments `a⊗a^T` for layer `i` |
| `self.cache_valid` | `bool` | Scalar | Flag indicating if cache is valid |

### Prediction (Full Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.cached_means_predict` | `list[np.ndarray]` | Length `L+1` | Cached prediction mean activations |
| `self.cached_means_predict[i]` | `np.ndarray` | Shape `(N_pred, D[i], 1)` | Mean activations for layer `i` |
| `self.cached_aats_predict` | `list[np.ndarray]` | Length `L+1` | Cached prediction second moments |
| `self.cached_aats_predict[i]` | `np.ndarray` | Shape `(N_pred, D[i], D[i])` | Second moments for layer `i` |
| `self.cache_valid_predict` | `bool` | Scalar | Flag for prediction cache validity |

### Prediction (Pruned Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.cached_means_predict[i]` | `np.ndarray` | Shape `(N_pred, n_active[i], 1)` | Pruned mean activations |
| `self.cached_aats_predict[i]` | `np.ndarray` | Shape `(N_pred, n_active[i], n_active[i])` | Pruned second moments |

---

## Sparsification Results

### Sparsification Outputs
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `active_weights` | `list[np.ndarray]` | Length `L+1` | Boolean masks for active weights |
| `active_weights[k]` | `np.ndarray` | Shape `(D[k+1], D[k])` | Boolean mask for layer `k` weights |
| `active_neurons` | `list[np.ndarray]` | Length `L+2` | Indices of active neurons |
| `active_neurons[0]` | `np.ndarray` | Shape `(n_active[0],)` | Active input feature indices |
| `active_neurons[k]` (k=1..L) | `np.ndarray` | Shape `(n_active[k],)` | Active hidden neuron indices |
| `active_neurons[L+1]` | `np.ndarray` | Shape `(D[L+1],)` | Output neurons (all kept) = `np.arange(D[L+1])` |
| `W_stars` | `list[np.ndarray]` | Length `L+1` | Sparsified weight matrices |
| `W_stars[k]` | `np.ndarray` | Shape `(D[k+1], D[k])` | Sparsified weights (zeros where inactive) |

---

## Input Data Handling

### Training Data (Full Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `self.x` | `np.ndarray` | Shape `(N, D_in, 1)` | Training input data (reshaped) |
| `self.y` | `np.ndarray` | Shape `(N, D_out)` | Training output data |

### Prediction Data (Full Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `x_for_pred` | `np.ndarray` | Shape `(N_pred, D_in)` | Original prediction input |
| `x_new` | `np.ndarray` | Shape `(N_pred, D_in, 1)` | Reshaped prediction input |

### Prediction Data (Pruned Model)
| Parameter | Type | Size | Description |
|-----------|------|------|-------------|
| `x_for_pred` | `np.ndarray` | Shape `(N_pred, D_in)` | Original prediction input |
| `x_for_pred[:, active_neurons[0]]` | `np.ndarray` | Shape `(N_pred, n_active[0])` | Filtered to active features |
| `x_new` | `np.ndarray` | Shape `(N_pred, n_active[0], 1)` | Reshaped pruned input |

---

## Dimension Relationships

### Full Model Layer Dimensions
```
Layer 0 (Input):     D[0] = D_in
Layer 1 (Hidden):    D[1] = D_H
Layer 2 (Hidden):    D[2] = D_H
...
Layer L (Hidden):    D[L] = D_H
Layer L+1 (Output):  D[L+1] = D_out
```

### Pruned Model Layer Dimensions
```
Layer 0 (Input):     n_active[0] ≤ D[0]
Layer 1 (Hidden):    n_active[1] ≤ D[1]
Layer 2 (Hidden):    n_active[2] ≤ D[2]
...
Layer L (Hidden):    n_active[L] ≤ D[L]
Layer L+1 (Output):  n_active[L+1] = D[L+1]  (all output neurons kept)
```

### Weight Matrix Dimensions
```
Full Model:
  m_h[k]:     (D[k+1], 1, D[k]+1)
  big_b[k]:   (D[k+1], D[k]+1, D[k]+1)
  
Pruned Model:
  m_h_pruned[k]:     (n_active[k+1], 1, n_active[k]+1)
  big_b_pruned[k]:   (n_active[k+1], n_active[k]+1, n_active[k]+1)
```

### Activation Parameter Dimensions
```
Full Model (Training):
  rho[k]:     (N, D[k+1])
  M[k][n]:    (D[k+1], D[k])
  b[k][n]:    (D[k+1], 1)
  S[k]:       (N, D[k+1], D[k+1])

Full Model (Prediction):
  rho_predict[k]:     (N_pred, D[k+1])
  M_predict[k][n]:    (D[k+1], D[k])
  b_predict[k][n]:    (D[k+1], 1)
  S_predict[k]:       (N_pred, D[k+1], D[k+1])

Pruned Model (Prediction):
  rho_predict[k]:     (N_pred, n_active[k+1])
  M_predict[k][n]:    (n_active[k+1], n_active[k])
  b_predict[k][n]:    (n_active[k+1], 1)
  S_predict[k]:       (N_pred, n_active[k+1], n_active[k+1])
```

---

## Matrix Operation Dimensions

### Forward Pass (Full Model)
```
Layer k activation:
  Input:  a[k-1]      (D[k-1], 1)
  Bias:   b[k][n]     (D[k], 1)
  Weight: M[k][n]     (D[k], D[k-1])
  Output: a[k]        (D[k], 1) = b[k][n] + M[k][n] @ a[k-1]

Second moment:
  Input:  aa^T[k-1]   (D[k-1], D[k-1])
  Cov:    S[k][n]     (D[k], D[k])
  Output: aa^T[k]     (D[k], D[k])
```

### Forward Pass (Pruned Model)
```
Layer k activation:
  Input:  a[k-1]      (n_active[k-1], 1)
  Bias:   b[k][n]     (n_active[k], 1)
  Weight: M[k][n]     (n_active[k], n_active[k-1])
  Output: a[k]        (n_active[k], 1) = b[k][n] + M[k][n] @ a[k-1]

Second moment:
  Input:  aa^T[k-1]   (n_active[k-1], n_active[k-1])
  Cov:    S[k][n]     (n_active[k], n_active[k])
  Output: aa^T[k]     (n_active[k], n_active[k])
```

---

## Summary Table: Key Parameter Transformations

| Parameter | Full Size | Pruned Size | Compression |
|-----------|-----------|-------------|-------------|
| Input features | `D[0]` | `n_active[0]` | `1 - n_active[0]/D[0]` |
| Hidden layer k | `D[k]` | `n_active[k]` | `1 - n_active[k]/D[k]` |
| Output layer | `D[L+1]` | `D[L+1]` | None (kept) |
| Weights m_h[k] | `(D[k+1], 1, D[k]+1)` | `(n_active[k+1], 1, n_active[k]+1)` | Parameter reduction |
| Activations M[k][n] | `(D[k+1], D[k])` | `(n_active[k+1], n_active[k])` | Computation reduction |

---

## Notes

1. **Singleton Dimension**: All weight means (`m_h`, `m_o`) include a middle singleton dimension (size 1) for consistency.

2. **Bias Indexing**: In all weight matrices, index `[:, 0, 0]` accesses biases, and `[:, 0, 1:]` accesses weights.

3. **List vs Array Structure**:
   - `M_predict` and `b_predict` are **lists of lists** (outer list = layers, inner list = samples)
   - `S_predict` is a **list of arrays** (list = layers, array includes all samples)

4. **Output Layer**: In the pruned model, all output neurons are retained (`n_active[L+1] = D[L+1]`), but connections only come from active hidden neurons.

5. **Cache Management**: Cache must be invalidated (`cache_valid = False`) whenever parameters are updated.

6. **Sample Size**: When `sample_size` is specified, stochastic training uses a subset of data, affecting dimensions of activation-related parameters.
