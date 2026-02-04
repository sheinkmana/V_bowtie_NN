
import numpy as np
from typing import Dict, Any, Union, Callable, List
import time
import pandas as pd
from datetime import datetime
import json
from VBNN_model.training_vbnn import VBNN_improving, VBNN_SVI_improving
from VBNN_model.masking_training_vbnn import VBNN_Iterative_CAVI, VBNN_Iterative_SVI
from config import ExperimentConfig, SVIExperimentConfig
from VBNN_model.utils import RMSE, NL, empirical_coverage_quantile
from pathlib import Path
from dataclasses import dataclass, asdict, field


class VBNNRunner:
    """Wrapper class for training and evaluating VBNN models (both CAVI and SVI)."""
    
    def __init__(self, config: Union['ExperimentConfig', 'SVIExperimentConfig'], dataset: Any):
        """
        Initialize VBNN runner.
        
        Args:
            config: Experiment configuration (ExperimentConfig for CAVI or SVIExperimentConfig for SVI)
            dataset: Pre-loaded dataset (optional). If None, will load from config
        """
        self.config = config
        self.model = None
        self.dataset = dataset
        self.results = {}
        self.is_svi = hasattr(config.training, 'sample_size')  # Detect SVI vs CAVI
    
    
    def train(self) -> Dict[str, Any]:        
        # Initialize model based on type
        print(f"Initializing {self.config.algorithm} model...")
        start_time = time.time()
        # In runner.py

        # 1. SVI Logic
        if self.is_svi:
            assert isinstance(self.config, SVIExperimentConfig)
            
            # Check for Iterative Pruning
            if getattr(self.config.training, 'iterative_pruning', False):
                print("Using VBNN_Iterative_SVI strategy.")
                ModelClass = VBNN_Iterative_SVI
            else:
                ModelClass = VBNN_SVI_improving

            self.model = ModelClass(
                x=self.dataset.X_train,
                y=self.dataset.y_train,
                D_H=self.config.model.D_H,
                L=self.config.model.L,
                T=self.config.model.T,
                wb_mode=self.config.model.wb_mode,
                big_S=self.config.model.big_S,
                sample_size=self.config.training.sample_size,
                big_B=self.config.model.big_B,
                beta_eta_h_prior=self.config.model.beta_eta_h_prior,
            )
            
            # Prepare kwargs based on whether it's iterative or not
            train_kwargs = {
                'epochs': self.config.training.epochs,
                'forgrate': self.config.training.forgetting_rate,
                'EM_step': self.config.training.EM_step,
                'rate_local': self.config.training.rate_local
            }
            
            if getattr(self.config.training, 'iterative_pruning', False):
                train_kwargs.update({
                    'pruning_interval': self.config.training.pruning_interval,
                    'pruning_alpha': self.config.training.pruning_alpha
                })

            print("\nTraining SVI model...")
            self.model.svi_alg(**train_kwargs)

        # 2. CAVI Logic
        else:
            assert isinstance(self.config, ExperimentConfig)
            
            # Check for Iterative Pruning
            if getattr(self.config.training, 'iterative_pruning', False):
                print("Using VBNN_Iterative_CAVI strategy.")
                ModelClass = VBNN_Iterative_CAVI
            else:
                ModelClass = VBNN_improving

            self.model = ModelClass(
                x=self.dataset.X_train,
                y=self.dataset.y_train,
                D_H=self.config.model.D_H,
                L=self.config.model.L,
                T=self.config.model.T,
                wb_mode=self.config.model.wb_mode,
                big_S=self.config.model.big_S,
                big_B=self.config.model.big_B,
                beta_eta_h_prior=self.config.model.beta_eta_h_prior,
            )
            
            # Prepare kwargs
            train_kwargs = {
                'epochs': self.config.training.epochs,
                'rate': self.config.training.rate,
                'EM_step': self.config.training.EM_step
            }

            if getattr(self.config.training, 'iterative_pruning', False):
                train_kwargs.update({
                    'pruning_interval': self.config.training.pruning_interval,
                    'pruning_alpha': self.config.training.pruning_alpha
                })
            
            print("\nTraining CAVI model...")
            self.model.algorithm(**train_kwargs)
        if self.is_svi:

            assert isinstance(self.config, SVIExperimentConfig)

            self.model = VBNN_SVI_improving(
                x=self.dataset.X_train,
                y=self.dataset.y_train,
                D_H=self.config.model.D_H,
                L=self.config.model.L,
                T=self.config.model.T,
                wb_mode=self.config.model.wb_mode,
                big_S=self.config.model.big_S,
                sample_size=self.config.training.sample_size,
                big_B=self.config.model.big_B,
                beta_eta_h_prior=self.config.model.beta_eta_h_prior,
            )
            
            # Train SVI model
            print("\nTraining SVI model...")
            self.model.svi_alg(
                epochs=self.config.training.epochs,
                forgrate=self.config.training.forgetting_rate,
                EM_step=self.config.training.EM_step,
                rate_local=self.config.training.rate_local
            )
        else:

            assert isinstance(self.config, ExperimentConfig)
            self.model = VBNN_improving(
                x=self.dataset.X_train,
                y=self.dataset.y_train,
                D_H=self.config.model.D_H,
                L=self.config.model.L,
                T=self.config.model.T,
                wb_mode=self.config.model.wb_mode,
                big_S=self.config.model.big_S,
                big_B=self.config.model.big_B,
                beta_eta_h_prior=self.config.model.beta_eta_h_prior,
            )
            
            # Train CAVI model
            print("\nTraining CAVI model...")
            self.model.algorithm(
                epochs=self.config.training.epochs,
                rate=self.config.training.rate,
                EM_step=self.config.training.EM_step
            )
        
        training_time = time.time() - start_time
        
        self.results['training_time'] = training_time
        self.results['epochs_completed'] = self.model.epoch_no
        self.results['final_elbo'] = self.model.elbo_total[-1] if self.model.elbo_total else None
        self.results['algorithm'] = self.config.algorithm
        
        print(f"\n{self.config.algorithm} training completed in {training_time:.2f} seconds")
        print(f"Epochs: {self.model.epoch_no}")
        if self.results['final_elbo']:
            print(f"Final ELBO: {self.results['final_elbo']:.4f}")
        
        return self.results
    

    def predict(self) -> Dict[str, Any]:
        """
        Make predictions using configuration.
        
        Supports two modes:
        1. Sparse (Pruned): Converts model to sparse lists, skipping zero weights.
        2. Dense (Normal): Uses full matrix multiplication (zeros are included in calc).
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("\nMaking predictions.")
        
        # 1. Sparse Prediction (Pruned Architecture)
        if self.config.prediction.sparse:
            
            # This method calculates the sparsity pattern (or uses the existing zeros 
            # from iterative pruning), converts matrices to ragged lists, and predicts.
            self.model.sparse_predict_pruned(
                x_for_pred=self.dataset.X_test,
                epochs_pred=self.config.prediction.epochs_pred,
                alpha=self.config.prediction.alpha,
                rate_pred=self.config.prediction.rate_pred
            )
            
        # 2. Normal Prediction (Dense Architecture)
        else:
        
            
            # If iterative pruning was used, this still works fine. 
            # The weights are 0.0, so the math is correct, just less efficient 
            # than the sparse mode.
            self.model.predict(
                x_for_pred=self.dataset.X_test,
                epochs_pred=self.config.prediction.epochs_pred,
                rate_pred=self.config.prediction.rate_pred
            )

        # 3. Extract Results
        # Both methods populate 'prediction_mean' and 'var_tot' attributes
        predictions = self.model.prediction_mean
        variance = self.model.var_tot
        
        # 4. Calculate Metrics
        rmse = RMSE(self.dataset.y_test, predictions)
        nll = NL(self.dataset.y_test, 
                 predictions.reshape(self.dataset.y_test.shape), 
                 np.sqrt(variance).reshape(self.dataset.y_test.shape))
        coverage = empirical_coverage_quantile(self.dataset.y_test, predictions, np.sqrt(variance))

        self.results.update({
            'test_rmse': rmse,
            'test_nll': nll,
            'test_coverage': coverage,
            'predictions': predictions,
            'variance': variance,
            'y_test': self.dataset.y_test
        })
        
        print("Test Results:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  NLL:  {nll:.4f}")
        print(f"  Coverage:   {coverage:.4f}")
        
        return self.results



    
    def save_results(self, filepath: str):
        """Save results to file."""
        import pickle
        
        save_dict = {
            'config': self.config,
            'results': self.results,
            'model_state': {
                'epoch_no': self.model.epoch_no if self.model else None,
                'elbo_total': self.model.elbo_total if self.model else None,
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"\nResults saved to {filepath}")



class DatasetSplitter:
    """Wrapper to generate multiple random splits."""
    
    def __init__(
        self,
        dataset_loader_fn: Callable,
        n_splits: int = 10,
        base_random_state: int = 42,
        **loader_kwargs
    ):
        self.dataset_loader_fn = dataset_loader_fn
        self.n_splits = n_splits
        self.base_random_state = base_random_state
        self.loader_kwargs = loader_kwargs
        
    def _load_split(self, split_id: int):
        return self.dataset_loader_fn(
            random_state=self.base_random_state + split_id,
            **self.loader_kwargs
        )
    
    def get_split(self, split_id: int):
        return self._load_split(split_id)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get metadata about the dataset."""
        dataset = self._load_split(0)
        return {
            'n_features': dataset.n_features,
            'n_train': dataset.n_train,
            'n_test': dataset.n_test,
            'feature_names': getattr(dataset, 'feature_names', []),
            'target_name': getattr(dataset, 'target_name', 'target')
        }

# -----------------------------------------------------------
# 3. Multi-Split Runner
# -----------------------------------------------------------
class MultiSplitRunner:
    """Runs experiment across multiple random splits for a SINGLE dataset."""
    
    def __init__(
        self, 
        config: Union['ExperimentConfig', 'SVIExperimentConfig'], 
        splitter: DatasetSplitter
    ):
        self.config = config
        self.splitter = splitter
        self.raw_results: List[Dict[str, Any]] = []
        self.summary: Dict[str, Dict[str, float]] = {}
        
    def run(self, verbose: bool = True):
        if verbose:
            print(f"  Running {self.splitter.n_splits} splits...")
        
        for i in range(self.splitter.n_splits):
            dataset = self.splitter.get_split(i)
            runner = VBNNRunner(self.config, dataset)
            
            train_res = runner.train()
            pred_res = runner.predict()
            
            split_result = {
                'split_id': i,
                **train_res,
                **pred_res,
            }
            
            # Lightweight storage: remove heavy arrays
            lightweight_result = {k: v for k, v in split_result.items() 
                                    if k not in ['predictions', 'variance', 'y_test', 'W_sparse', 'connections']}
            
            self.raw_results.append(lightweight_result)
        
                
        self._compute_summary()
            
    def _compute_summary(self):
        if not self.raw_results:
            return

        metrics_of_interest = [
            'test_rmse', 'test_nll', 'test_coverage', 
            'training_time', 'final_elbo', 'epochs_completed'
        ]
        


        self.summary = {}
        
        for metric in metrics_of_interest:
            raw_values = [r.get(metric) for r in self.raw_results if r.get(metric) is not None]
            
            if raw_values:
                values = np.array(raw_values, dtype=np.float64)
                self.summary[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

# -----------------------------------------------------------
# 4. Benchmark Result Dataclass & Runner
# -----------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Data structure to hold aggregated results for a single dataset."""
    dataset_name: str
    model_name: str
    inference_method: str
    temperature: float
    iterative_pruning: bool 
    pruning_interval: int 
    pruning_alpha: float
    big_S: float
    big_B: float
    no_layers: int
    hid_dim: int
    beta_h_prior: float
    
    # Metrics
    rmse_mean: float
    rmse_std: float
    nll_mean: float
    nll_std: float
    coverage_mean: float
    coverage_std: float
    avg_train_time: float
    
    # Metadata
    n_splits: int
    n_train_samples: int
    n_test_samples: int
    n_features: int
    timestamp: str
    
    # Raw split data
    split_results: List[Dict[str, Any]] = field(default_factory=list)

class BenchmarkRunner:
    """
    Iterates over datasets, runs experiments, and saves results in JSON/CSV formats.
    """
    
    def __init__(
        self, 
        config: Union['ExperimentConfig', 'SVIExperimentConfig'], 
        datasets: Dict[str, DatasetSplitter],
        output_dir: str = "results"
    ):
        self.config = config
        self.datasets = datasets
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_results: List[BenchmarkResult] = []
        
    def run_all(self):
        """Run benchmark on all datasets and save results."""
        print(f"Starting Benchmark on {len(self.datasets)} datasets.")
        print(f"Method: {self.config.algorithm} | Mode: {self.config.model.wb_mode}")
        print(f"Output Directory: {self.output_dir.resolve()}")
        
        for name, splitter in self.datasets.items():
            print(f"\nProcessing Dataset: {name.lower()}")
            
            # 1. Run Multi-Split Experiment
            ms_runner = MultiSplitRunner(self.config, splitter)
            ms_runner.run(verbose=True)
            
            # 2. Extract Data
            summary = ms_runner.summary
            meta = splitter.get_dataset_info()
            
            # Helper to safely get stats (returns 0.0 or nan if missing)
            def get_val(metric, stat):
                return summary.get(metric, {}).get(stat, np.nan)

            # 3. Create Result Object
            result = BenchmarkResult(
                dataset_name=name,
                model_name=self.config.model.wb_mode,
                inference_method=self.config.algorithm,
                temperature=self.config.model.T,
                iterative_pruning=getattr(self.config.training, 'iterative_pruning', False),
                pruning_interval=getattr(self.config.training, 'pruning_interval', 0),
                pruning_alpha=getattr(self.config.training, 'pruning_alpha', 0.0),
                big_S=self.config.model.big_S,
                big_B=self.config.model.big_B,
                beta_h_prior=self.config.model.beta_eta_h_prior,
                no_layers=self.config.model.L,
                hid_dim=self.config.model.D_H,
                rmse_mean=get_val('test_rmse', 'mean'),
                rmse_std=get_val('test_rmse', 'std'),
                nll_mean=get_val('test_nll', 'mean'),
                nll_std=get_val('test_nll', 'std'),
                coverage_mean=get_val('test_coverage', 'mean'),
                coverage_std=get_val('test_coverage', 'std'),
                avg_train_time=get_val('training_time', 'mean'),
                
                
                n_splits=splitter.n_splits,
                n_train_samples=meta.get('n_train', 0),
                n_test_samples=meta.get('n_test', 0),
                n_features=meta.get('n_features', 0),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                
                split_results=ms_runner.raw_results
            )
            
            self.all_results.append(result)
            
            # 4. Save Individual Result Immediately
            self._save_individual_result(result)
            
        # 5. Save Consolidated Results
        self._save_consolidated_results()

    def _save_individual_result(self, results: BenchmarkResult):
        """Save JSON for a single dataset immediately after processing."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{results.dataset_name}_{results.model_name}_{results.inference_method}"
        
        agg_file = self.output_dir / f"{base_name}_{timestamp}_aggregated.json"
        
        with open(agg_file, 'w') as f:
            # Convert dataclass to dict
            results_dict = asdict(results)
            # Ensure serialization of any numpy types that might have slipped through
            json.dump(results_dict, f, indent=2, default=self._json_serializer)
            
        print(f"  -> Saved individual result to: {agg_file.name}")

    def _save_consolidated_results(self):
        """Save CSV summary and full JSON details for all datasets."""
        if not self.all_results:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Create DataFrame for CSV
        rows = []
        for result in self.all_results:
            row = {
                'dataset': result.dataset_name,
                'model': result.model_name,
                'method': result.inference_method,
                'temperature': result.temperature,
                'iterative_pruning': result.iterative_pruning,
                'pruning_interval': result.pruning_interval,
                'pruning_alpha': result.pruning_alpha,
                'big_S': result.big_S,
                'big_B': result.big_B,
                'beta_eta_h_prior': result.beta_h_prior,
                'no_layers': result.no_layers,
                'hid_dim': result.hid_dim,
                'rmse_mean': result.rmse_mean,
                'rmse_std': result.rmse_std,
                'nll_mean': result.nll_mean,
                'nll_std': result.nll_std,
                'coverage_mean': result.coverage_mean,
                'coverage_std': result.coverage_std,
                'avg_train_time': result.avg_train_time,
                'n_total_splits': result.n_splits,
                'n_train': result.n_train_samples,
                'n_test': result.n_test_samples,
                'n_features': result.n_features,
                'timestamp': result.timestamp
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_file = self.output_dir / f"all_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nSaved consolidated CSV to: {csv_file}")
        
        # 2. Save Full JSON
        json_file = self.output_dir / f"all_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(
                [asdict(r) for r in self.all_results], 
                f, 
                indent=2, 
                default=self._json_serializer
            )
        print(f"Saved consolidated JSON to: {json_file}")

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Helper to serialize numpy types, JAX arrays, and others for JSON."""
        # 1. Handle Numpy scalars
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
            
        # 2. Handle Numpy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
            
        # 3. Handle Datetimes
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
            
        # 4. Handle JAX arrays (ArrayImpl) and other array-likes
        # JAX arrays don't inherit from np.ndarray, but they have a .tolist() method.
        if hasattr(obj, 'tolist') and callable(obj.tolist):
            return obj.tolist()
            
        # 5. Last resort: try converting to numpy array (catches custom array wrappers)
        if hasattr(obj, '__array__'):
            return np.array(obj).tolist()

        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# import pandas as pd
# from config import ExperimentConfig
# from runner import BenchmarkRunner, DatasetSplitter
# from data.data import DatasetLoader 
# from config import ExperimentConfig, ModelConfig, TrainingConfig, PredictionConfig, SVIExperimentConfig, SVITrainingConfig

# # 1. Define your Configuration
# config = ExperimentConfig(
#     model=ModelConfig(D_H=10, L=2, T=0.1, wb_mode='laplace', big_S=0.01),
#     training=TrainingConfig(epochs=2, rate=1e-5, EM_step=False),
#     prediction=PredictionConfig(epochs_pred=2, rate_pred=1e-4)
# )
# # config.model.L = 2
# # config.model.D_H = 50
# # config.training.epochs = 100
# # config.prediction.sparse = False # Set to True if you want sparsity metrics





# # 2. Define your Datasets Dictionary (as provided in your prompt)
# datasets = {
#     'boston': DatasetSplitter(
#         DatasetLoader.load_boston,
#         n_splits=10,
#         filepath='data/boston.txt',
#         test_size=0.1,
#         scale=True
#     )
# }

# # 3. Initialize and Run Benchmark
# benchmark = BenchmarkRunner(config, datasets)
# benchmark.run_all()
