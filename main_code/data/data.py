import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from typing import Optional


@dataclass
class Dataset:
    """Container for train/test split with preprocessing."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: Optional[preprocessing.StandardScaler] = None
    feature_names: Optional[list] = None
    target_name: Optional[str] = None
    
    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]
    
    @property
    def n_train(self) -> int:
        return self.X_train.shape[0]
    
    @property
    def n_test(self) -> int:
        return self.X_test.shape[0]
    
    def summary(self) -> str:
        return (f"Dataset Summary:\n"
                f"  Train samples: {self.n_train}\n"
                f"  Test samples: {self.n_test}\n"
                f"  Features: {self.n_features}\n"
                f"  Feature names: {self.feature_names}\n"
                f"  Target: {self.target_name}")


class DatasetLoader:
    """Factory class for loading different datasets."""
    
    @staticmethod
    def load_boston(
        filepath: str = 'data/boston.txt',
        test_size: float = 0.1,
        random_state: int = 42,
        scale: bool = True
    ) -> Dataset:
        """
        Load Boston Housing dataset.
        
        Args:
            filepath: Path to the data file
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            scale: Whether to standardize features
            
        Returns:
            Dataset object with train/test split
        """
        # Column names
        feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
                        'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        target_name = 'MEDV'
        
        # Load data
        df = pd.read_csv(filepath, header=None, delimiter=r"\s+")
        df.columns = feature_names + [target_name]
        
        # Split features and target
        X = df[feature_names].values
        y = df[[target_name]].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
      
        
        # Optional scaling
        scaler = None
        if scale:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        return Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=feature_names,
            target_name=target_name
        )
    
    @staticmethod
    def load_power(
        filepath: str = 'data/power.csv',
        test_size: float = 0.1,
        random_state: int = 42,
        scale: bool = True
    ) -> Dataset:
        """
        Load Cycle power plant dataset.
        
        Args:
            filepath: Path to the data file
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            scale: Whether to standardize features
            
        Returns:
            Dataset object with train/test split
        """
        # Column names
        feature_names = ['AT','V','AP','RH']
        target_name = 'PE'
        
        # Load data
        df = pd.read_csv(filepath)
        # df.columns = feature_names + [target_name]
        
        # Split features and target
        X = df[feature_names].values
        y = df[[target_name]].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Optional scaling
        scaler = None
        if scale:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        return Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=feature_names,
            target_name=target_name
        )
    
    @staticmethod
    def load_energy(
        filepath: str = 'data/energy.csv',
        test_size: float = 0.1,
        random_state: int = 42,
        scale: bool = True
    ) -> Dataset:
        """
        Load Energy  dataset.
        
        Args:
            filepath: Path to the data file
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            scale: Whether to standardize features
            
        Returns:
            Dataset object with train/test split
        """
        # Column names
        feature_names = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
        target_name = 'Y1'
        
        # Load data
        df = pd.read_csv(filepath)
        # df = df[df.columns[:-1]]
        # df.columns = feature_names + [target_name]
        
        # Split features and target
        X = df[feature_names].values
        y = df[[target_name]].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Optional scaling
        scaler = None
        if scale:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        return Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=feature_names,
            target_name=target_name
        )
    
    @staticmethod
    def load_nasa(
        filepath: str = 'data/nasa.csv',
        test_size: float = 0.1,
        random_state: int = 42,
        scale: bool = True
    ) -> Dataset:
        """
        Load nasa dataset.
        
        Args:
            filepath: Path to the data file
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            scale: Whether to standardize features
            
        Returns:
            Dataset object with train/test split
        """
        # Column names
        feature_names = ['mach','alpha', 'beta']
        target_name = 'lift'
        
        # Load data
        df = pd.read_csv(filepath)
        # df.columns = feature_names + [target_name]
        
        # Split features and target
        X = df[feature_names].values
        y = df[[target_name]].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Optional scaling
        scaler = None
        if scale:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        return Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=feature_names,
            target_name=target_name
        )
    
    

    @staticmethod
    def load_concrete(
        filepath: str = 'data/concrete.txt',
        test_size: float = 0.1,
        random_state: int = 42,
        scale: bool = True
    ) -> Dataset:
        """
        Concrete dataset.
        
        Args:
            filepath: Path to the data file
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            scale: Whether to standardize features
            
        Returns:
            Dataset object with train/test split
        """
        # Column names
        feature_names = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']
        target_name = 'Concrete compressive strength'
        
        # Load data
        df = pd.read_csv(filepath, header=None, sep=r'\s+')
        df.columns = feature_names + [target_name]
        
        # Split features and target
        X = df[feature_names].values
        y = df[[target_name]].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Optional scaling
        scaler = None
        if scale:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        return Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=feature_names,
            target_name=target_name
        )
    

    @staticmethod
    def load_yacht(
        filepath: str = 'data/yacht.txt',
        test_size: float = 0.1,
        random_state: int = 42,
        scale: bool = True
    ) -> Dataset:
        """
        Yacht hydrodinamics dataset.
        
        Args:
            filepath: Path to the data file
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            scale: Whether to standardize features
            
        Returns:
            Dataset object with train/test split
        """
        # Column names
        feature_names = ['0','1','2','3','4','5']
        target_name = '6'
        
        # Load data
        df = pd.read_csv(filepath, header=None, sep=r'\s+')
        df.columns = feature_names + [target_name]
        
        # Split features and target
        X = df[feature_names].values
        y = df[[target_name]].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Optional scaling
        scaler = None
        if scale:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        return Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=feature_names,
            target_name=target_name
        )
    
    # @staticmethod
    # def load_california_housing(
    #     test_size: float = 0.1,
    #     random_state: int = 42,
    #     scale: bool = True
    # ) -> Dataset:
    #     """Load California Housing dataset from sklearn."""
    #     from sklearn.datasets import fetch_california_housing
        
    #     data = fetch_california_housing()
    #     X, y = data.data, data.target.reshape(-1, 1)
        
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=test_size, random_state=random_state
    #     )
        
    #     scaler = None
    #     if scale:
    #         scaler = preprocessing.StandardScaler().fit(X_train)
    #         X_train = scaler.transform(X_train)
    #         X_test = scaler.transform(X_test)
        
    #     return Dataset(
    #         X_train=X_train,
    #         X_test=X_test,
    #         y_train=y_train,
    #         y_test=y_test,
    #         scaler=scaler,
    #         feature_names=list(data.feature_names),
    #         target_name='MedHouseVal'
    #     )
    
    # @staticmethod
    # def load_diabetes(
    #     test_size: float = 0.1,
    #     random_state: int = 42,
    #     scale: bool = True
    # ) -> Dataset:
    #     """Load Diabetes dataset from sklearn."""
    #     from sklearn.datasets import load_diabetes
        
    #     data = load_diabetes()
    #     X, y = data.data, data.target.reshape(-1, 1)
        
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X, y, test_size=test_size, random_state=random_state
    #     )
        
    #     scaler = None
    #     if scale:
    #         scaler = preprocessing.StandardScaler().fit(X_train)
    #         X_train = scaler.transform(X_train)
    #         X_test = scaler.transform(X_test)
        
    #     return Dataset(
    #         X_train=X_train,
    #         X_test=X_test,
    #         y_train=y_train,
    #         y_test=y_test,
    #         scaler=scaler,
    #         feature_names=list(data.feature_names),
    #         target_name='progression'
    #     )
    
    @staticmethod
    def load_custom_csv(
        filepath: str,
        feature_cols: list,
        target_col: str,
        test_size: float = 0.1,
        random_state: int = 42,
        scale: bool = True,
        **read_csv_kwargs
    ) -> Dataset:
        """
        Load custom CSV dataset.
        
        Args:
            filepath: Path to CSV file
            feature_cols: List of feature column names
            target_col: Target column name
            test_size: Proportion for test set
            random_state: Random seed
            scale: Whether to scale features
            **read_csv_kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Dataset object
        """
        df = pd.read_csv(filepath, **read_csv_kwargs)
        
        X = df[feature_cols].values
        y = df[[target_col]].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        scaler = None
        if scale:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        return Dataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=feature_cols,
            target_name=target_col
        )

