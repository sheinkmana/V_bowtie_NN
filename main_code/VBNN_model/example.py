import os
from config import ExperimentConfig, ModelConfig, TrainingConfig, PredictionConfig
from runner import BenchmarkRunner, DatasetSplitter
from data.data import DatasetLoader 

def main():
    # ---------------------------------------------------------
    # 1. Define Configuration
    # ---------------------------------------------------------
    config = ExperimentConfig(
        model=ModelConfig(
            D_H=50,                 # Hidden layer size
            L=2,                    # Number of hidden layers
            T=0.001,                # Temperature
            wb_mode='spikeslab',    # Initialization mode
            big_S=0.001,            # Prior variance for S
            big_B=0.0001,           # Prior variance for B
            beta_eta_h_prior=0.0001 # Noise prior
        ),
        training=TrainingConfig(
            epochs=100,             # Total training epochs
            rate=1e-5,              # Convergence rate
            EM_step=False,          # EM update for hyperparameters
            
            # --- Iterative Pruning Settings ---
            iterative_pruning=True, 
            pruning_interval=20,    # Prune every 20 epochs
            pruning_alpha=0.01      # FDR threshold (lower = stricter)
        ),
        prediction=PredictionConfig(
            epochs_pred=50, 
            rate_pred=1e-4,
            
            # --- Prediction Mode ---
            # True = Use sparse_predict_pruned() (ragged lists, ignores zeros)
            # False = Use predict() (dense matrix mult, zeros included)
            sparse=True,            
            alpha=0.01              
        )
    )

    # ---------------------------------------------------------
    # 2. Define Dataset
    # ---------------------------------------------------------
    # Ensure the data file exists relative to where you run the script
    data_path = 'data/energy.csv'
    
    if not os.path.exists(data_path):
        print(f"Warning: '{data_path}' not found. Please ensure the CSV is in the data folder.")

    datasets = {
        'energy': DatasetSplitter(
            DatasetLoader.load_energy, # Uses your existing static method
            n_splits=5,                # Number of random splits to run
            filepath=data_path,
            test_size=0.1,
            scale=True                 # Standardize features (recommended for VBNN)
        )
    }

    # ---------------------------------------------------------
    # 3. Run Benchmark
    # ---------------------------------------------------------
    print(f"--- Starting Benchmark on Energy Dataset ---")
    print(f"Model: {config.model.wb_mode}")
    print(f"Training: Iterative Pruning = {config.training.iterative_pruning}")
    print(f"Prediction: Sparse Mode = {config.prediction.sparse}")
    
    # Results will be saved to a new 'results_energy' folder
    benchmark = BenchmarkRunner(
        config=config, 
        datasets=datasets, 
        output_dir="results_energy"
    )
    
    benchmark.run_all()
    print("\n--- Benchmark Complete ---")
    print("Check the 'results_energy' folder for JSON/CSV reports.")

if __name__ == "__main__":
    main()