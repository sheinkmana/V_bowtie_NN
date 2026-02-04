This repository provides a Python implementation of the Variational Bow tie neural network (VBNN) which is described and derived in a [Variational Bayesian Bow tie Neural Networks with Shrinkage](https://arxiv.org/abs/2411.11132) paper. 
 ---
```
V_bowtie_NN/
├── main_code/
│   │
│   ├── VBNN_model/                  # CORE MODEL LOGIC
│   │   ├── __init__.py
│   │   ├── base_vbnn.py             # Base class 
│   │   ├── prediction_vbnn.py       # Prediction Mixin 
│   │   ├── sparse_vbnn.py           # Sparsity Mixin 
│   │   ├── mixer_vbnn.py            # Composition Root 
│   │   ├── training_vbnn.py         # Dense Training (CAVI & SVI algorithms)
│   │   ├── masking_training_vbnn.py # Iterative Pruning (Masking during training)
│   │   ├── training_sparse_vbnn.py  # Sparse Training 
│   │   └── utils.py                 # Math helpers
│   │
│   ├── data/                        # DATA MANAGEMENT
│   │   ├── __init__.py
│   │   └── data.py                  # Dataset loaders & preprocessing
│   │
│   ├── config.py                    # Configuration Dataclasses
│   ├── runner.py                    # Experiment & Benchmarking
│   ├── example.py            	     # Example
│   │
├── toy_example/	 # Older toy example
│   │
│   ├── VBNN_class.py 	# Old
│   ├── test.ipynb 	# Old
│   ├──helpers.py 	# Old
│   │
├── README.md
└── .gitignore
```
----
### The `main_code` folder contains the full VBNN model's code, as well as utilities for model running and benchmarking:
#### 1. Core Model Architecture
* **`base_vbnn.py` (`VBNNCore`)**
  * **Role:** The foundation of the model. It handles the `__init__` process.
  * **Key Responsibilities:**
    * **Architecture Setup:** Defines dimensions $D$ (layers) and $L$ (depth).
    * **Prior Initialization:** Sets up the hierarchical priors for the variance parameters ($\alpha, \beta$) and shrinkage parameters ($\nu, \delta$).
    * **Weight Initialization:** Implements various initialization strategies (`laplace`, `spikeslab`, `numpyro_init`, ` sparse_init`) to set starting values for the variational means (`m_h`, `m_o`).
    * **Forward Pass:** Implements `_compute_forward_pass` for dense matrices, caching the first and second moments of activations ($E[a]$ and $E[aa^T]$).
* **`mixer_vbnn.py` (`VBNNBase`)**
  * **Role:** Mixer class.
  * **Key Responsibilities:** It aggregates the functionality of `VBNNCore`, `VBNNPredictionMixin`, and `VBNNSparsityMixin` into a single class `VBNNBase`. This allows the training classes to inherit all capabilities (dense training, prediction, and pruning) in one go.

#### 2. Training Module
* **`training_vbnn.py`**
  * **Role:** Implements the training logic for the **Dense** (fully connected) network by maxmizing the Evidence Lower Bound **ELBO**.
  * **Classes:**
    * `VBNN_improving`: Implements **CAVI** (Coordinate Ascent Variational Inference). 
    * `VBNN_SVI_improving`: Implements **SVI** (Stochastic Variational Inference). 
    
#### 2. Prediction Module
* **`prediction_vbnn.py` (`VBNNPredictionMixin`)**
  * **Role:** Handles inference on test data.
  * **Key Responsibilities:**
    * **`predict`:** Runs the forward pass on new data to generate mean predictions.
    * **Uncertainty Quantification:** Calculates the total variance (`var_tot`), decomposing it into linear variance (`var_lin`, epistemic) and noise variance (aleatoric).

#### 3. Sparsity 
* **`sparify_vbnn.py` (`VBNNSparsityMixin`)**
  * **Role:** Implements the statistical pruning logic.
  * **Key Responsibilities:**
    * The `FDR` function calculates the estimated False Discovery Rate based on the signal-to-noise ratio of the variational posteriors (mean divided by standard deviation).
    * **`sparse_weights`:** Determines which weights to keep based on a desired FDR threshold ($\alpha$).
    * `mask_parameters`: Sets irrelevant weights to zero without deleting the nodes. Used for training with iterative pruning.
    * **`prune_parameters`:** Physically extracts the active parameters from the dense matrices to create the ragged structures used by `VBNN_SparseTraining` (see below). 
* **`masking_training_vbnn.py`** (`VBNN_Iterative_CAVI` or `VBNN_Iterative_SVI`)
  * **Role:** Implementation of the Iterative Pruning strategy (train, prune, traine, prune, repeat).
  * **Key Responsibilities:** It trains a dense network but periodically forces specific weights to zero during the training process.
  * 
#### 4. Utils and configs
These files provide the user interface for running experiments:
* **`config.py`**
* **`runner.py`**
* `example.py` is an example of using all of the above. 
* `utils.py` contains the low-level mathematical primitives required for the VBNN. 

--------
### The ` toy_example`  folder has the following older files:
- `VBNN_class.py` which contains the simplified code logic for the network. 
- `helpers.py` which contains utilities for easy training and testing.
- `test.ipynb` notebook which shows how to run VBNN on a simulated example.   
