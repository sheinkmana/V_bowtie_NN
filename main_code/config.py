from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
import json


@dataclass
class ModelConfig:
    """Configuration for VBNN model architecture and initialization."""
    D_H: int = 20
    """Hidden layer dimension (applied to all hidden layers)"""
    L: int = 2
    """Number of hidden layers"""
    T: float = 0.1
    """Temperature parameter for sigmoid approximation"""
    wb_mode: Literal['laplace', 'spikeslab', 'numpyro_init', 'sparse_init'] = 'laplace'
    """Weight initialization mode"""
    big_S: float = 0.001
    """Initial covariance scale for activations"""
    big_B: float = 0.0001
    "Parameters for eta_h"
    beta_eta_h_prior: float = 0.0001
    
    # # Prior hyperparameters
    # alpha_eta_h_prior: float = 2.0
    # alpha_eta_o_prior: float = 2.0
    # beta_eta_h_prior: float = 0.01
    # nu_tau_prior: float = -1.5
    # nu_psi_prior: float = -1.5
    
    def __post_init__(self):
        """Validate configuration."""
        if self.D_H <= 0:
            raise ValueError(f"D_H must be positive, got {self.D_H}")
        if self.L <= 0:
            raise ValueError(f"L must be positive, got {self.L}")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")
        if self.wb_mode not in ['laplace', 'spikeslab','numpyro_init', 'sparse_init']:
            raise ValueError(f"wb_mode must be 'laplace' or 'spikeslab' or 'numpyro_init' or 'sparse_init', got {self.wb_mode}")


@dataclass
class TrainingConfig:
    """Configuration for CAVI training algorithm."""
    epochs: int = 200
    """Maximum number of training epochs"""
    rate: float = 1e-5
    """Convergence rate threshold for early stopping"""
    EM_step: bool = False
    """Whether to use EM updates for delta parameters"""
    

@dataclass
class SVITrainingConfig:
    """Configuration for SVI (Stochastic Variational Inference) training."""
    epochs: int = 200
    """Maximum number of training epochs"""
    sample_size: int = 100
    """Mini-batch size for stochastic updates"""
    local_epochs: int = 40
    """Number of local optimization iterations per global update"""
    rate_local: float = 1e-4
    """Convergence rate for local parameter optimization"""
    forgetting_rate: float = 0.75
    """Forgetting rate for stochastic gradient updates (learning rate decay)"""
    update_a: bool = True
    """Update activation parameters"""
    update_rho: bool = True
    """Update rho (sigmoid) parameters"""
    update_weights: bool = True
    """Update weight parameters"""
    update_weightsout: bool = True
    """Update output layer weights"""
    update_eta: bool = True
    """Update precision (eta) parameters"""
    EM_step: bool = False
    """Whether to use EM updates for delta parameters"""

@dataclass
class PredictionConfig:
    """Configuration for prediction."""
    epochs_pred: int = 60
    """Number of epochs for prediction optimization"""
    rate_pred: float = 1e-4
    """Convergence rate threshold for prediction"""
    sparse: bool = False
    """Use sparse prediction"""
    alpha: float = 0.001
    """FDR level for sparsity (when sparse=True)"""
    

@dataclass
class ExperimentConfig:
    """Complete experiment configuration for CAVI."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    
    # Experiment metadata
    experiment_name: Optional[str] = None
    algorithm: str = "CAVI"
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclasses
        model_config = ModelConfig(**config_dict.pop('model'))
        training_config = TrainingConfig(**config_dict.pop('training'))
        prediction_config = PredictionConfig(**config_dict.pop('prediction'))
        
        return cls(
            model=model_config,
            training=training_config,
            prediction=prediction_config,
            **config_dict
        )
    

@dataclass
class SVIExperimentConfig:
    """Complete experiment configuration for SVI."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: SVITrainingConfig = field(default_factory=SVITrainingConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    
    # Experiment metadata
    experiment_name: Optional[str] = None
    algorithm: str = "SVI"
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SVIExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested dataclasses
        model_config = ModelConfig(**config_dict.pop('model'))
        training_config = SVITrainingConfig(**config_dict.pop('training'))
        prediction_config = PredictionConfig(**config_dict.pop('prediction'))
        
        return cls(
            model=model_config,
            training=training_config,
            prediction=prediction_config,
            **config_dict
        )