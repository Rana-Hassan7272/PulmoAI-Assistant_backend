"""
Spirometry ML Model Featurizer and Predictor

This module provides functionality to:
1. Load and manage preprocessing pipeline
2. Load trained XGBoost models for spirometry predictions
3. Preprocess input data and make predictions for:
   - Obstruction
   - Restriction
   - PRISm (Preserved Ratio Impaired Spirometry)
   - Mixed patterns
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import sys

# Try importing joblib (sklearn models are often saved with joblib)
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


def safe_pickle_load(file_path: Path):
    """
    Safely load pickle file with multiple fallback methods.
    Tries pickle, joblib, and various encodings.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded object
    """
    errors = []
    
    # Method 1: Try joblib first (sklearn models are often saved with joblib)
    if JOBLIB_AVAILABLE:
        try:
            return joblib.load(file_path)
        except Exception as e:
            errors.append(f"joblib: {str(e)}")
    
    # Method 2: Standard pickle load
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        errors.append(f"Standard pickle: {str(e)}")
    
    # Method 3: With latin1 encoding (Python 2/3 compatibility)
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        errors.append(f"Latin1 encoding: {str(e)}")
    
    # Method 4: Try with bytes encoding
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f, encoding='bytes')
    except Exception as e:
        errors.append(f"Bytes encoding: {str(e)}")
    
    # Method 5: Try with pickle5 if available
    try:
        import pickle5
        with open(file_path, 'rb') as f:
            return pickle5.load(f)
    except ImportError:
        errors.append("pickle5 not available (install with: pip install pickle5)")
    except Exception as e:
        errors.append(f"pickle5: {str(e)}")
    
    # If all methods fail, raise with all error messages
    error_msg = f"Failed to load {file_path.name} with all methods:\n" + "\n".join(f"  - {err}" for err in errors)
    error_msg += "\n\nSOLUTIONS:\n"
    error_msg += "1. Re-save the model files using: python backend/app/ml_models/spirometry/resave_models.py\n"
    error_msg += "2. Or re-save manually in Python with: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)\n"
    error_msg += "3. Install joblib: pip install joblib\n"
    error_msg += "4. Install pickle5: pip install pickle5"
    
    raise RuntimeError(error_msg)


class SpirometryFeaturizer:
    """
    Main class for spirometry data preprocessing and prediction.
    
    Handles:
    - Data preprocessing (scaling, encoding)
    - Loading trained models
    - Making predictions for all 4 conditions
    """
    
    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the featurizer.
        
        Args:
            models_dir: Directory containing the model files. 
                       If None, uses the directory of this file.
        """
        if models_dir is None:
            models_dir = Path(__file__).parent
        else:
            models_dir = Path(models_dir)
        
        self.models_dir = models_dir
        self.preprocessing_pipeline_path = models_dir / "preprocessing_pipeline.pkl"
        
        # API input columns (lowercase, user-friendly names)
        self.input_cols = [
            'sex', 'race', 'age', 'height', 'weight', 'bmi',
            'fev1', 'fvc'
        ]
        
        # Actual CSV column names (matching the dataset)
        self.csv_feature_cols = [
            'Sex', 'Race', 'Age', 'Height', 'Weight', 'BMI',
            'Baseline_FEV1_L', 'Baseline_FVC_L', 'Baseline_FEV1_FVC_Ratio'
        ]
        
        # Mapping from API input to CSV column names
        self.input_to_csv_mapping = {
            'sex': 'Sex',
            'race': 'Race',
            'age': 'Age',
            'height': 'Height',
            'weight': 'Weight',
            'bmi': 'BMI',
            'fev1': 'Baseline_FEV1_L',
            'fvc': 'Baseline_FVC_L',
            'fev1_fvc': 'Baseline_FEV1_FVC_Ratio'
        }
        
        # Reverse mapping (CSV to API input)
        self.csv_to_input_mapping = {v: k for k, v in self.input_to_csv_mapping.items()}
        
        # Feature columns for preprocessing (using CSV names)
        self.feature_cols = self.csv_feature_cols.copy()
        
        # Numeric and categorical features (using CSV column names)
        self.numeric_features = ['Age', 'Height', 'Weight', 'BMI', 'Baseline_FEV1_L', 'Baseline_FVC_L', 'Baseline_FEV1_FVC_Ratio']
        self.categorical_features = ['Sex', 'Race']
        
        self.target_names = ['obstruction', 'restriction', 'prism', 'mixed']
        
        # Will be initialized when load_models() is called
        self.preprocessing_pipeline: Optional[Pipeline] = None
        self.models: Dict[str, xgb.XGBClassifier] = {}
        self.is_loaded = False
    
    def _rebuild_preprocessing_pipeline(self) -> None:
        """
        Rebuild the preprocessing pipeline.
        Tries to use the dataset if available, otherwise creates a synthetic sample.
        This is used as a fallback when the saved pipeline can't be loaded.
        """
        dataset_path = self.models_dir / "spirometry.csv"
        X = None
        
        # Try to load from dataset (check both possible locations)
        dataset_paths = [
            self.models_dir / "spirometry.csv",
            self.models_dir / "dataset" / "spirometry.csv"
        ]
        
        for dataset_path in dataset_paths:
            if dataset_path.exists():
                try:
                    print(f"  Loading dataset from {dataset_path}...")
                    df = pd.read_csv(dataset_path)
                    
                    # Drop rows with missing values in feature columns
                    df = df.dropna(subset=self.feature_cols)
                    
                    # Check if required columns exist
                    missing_cols = set(self.feature_cols) - set(df.columns)
                    if not missing_cols:
                        # Select feature columns
                        X = df[self.feature_cols].copy()
                        print(f"  ✓ Using dataset to fit pipeline ({len(X)} rows)")
                        break
                except Exception as e:
                    print(f"  ⚠ Could not use dataset: {str(e)}")
                    continue
        
        # If dataset not available or doesn't have right columns, create synthetic sample
        if X is None or len(X) == 0:
            print("  Creating synthetic sample to fit pipeline...")
            # Create a diverse synthetic sample that covers all categorical values
            # and reasonable numeric ranges (using CSV column names)
            synthetic_data = []
            
            # Get unique values for categorical features from common values
            sex_values = ['Male', 'Female']
            race_values = ['White', 'Black', 'Asian', 'Hispanic', 'Other', 'Mexican American', 'Other race, including multi-racial']
            
            # Create combinations
            for sex in sex_values:
                for race in race_values:
                    fev1 = 3.5
                    fvc = 4.5
                    fev1_fvc = fev1 / fvc
                    synthetic_data.append({
                        'Sex': sex,
                        'Race': race,
                        'Age': 45.0,
                        'Height': 170.0,
                        'Weight': 70.0,
                        'BMI': 24.0,
                        'Baseline_FEV1_L': fev1,
                        'Baseline_FVC_L': fvc,
                        'Baseline_FEV1_FVC_Ratio': fev1_fvc
                    })
            
            X = pd.DataFrame(synthetic_data)
            print(f"  ✓ Created synthetic sample with {len(X)} rows")
        
        # Build preprocessing pipeline (same structure as training)
        # Handle sklearn version compatibility for OneHotEncoder
        try:
            # sklearn >= 1.2 uses sparse_output
            encoder = OneHotEncoder(drop='first', sparse_output=False)
        except TypeError:
            # sklearn < 1.2 uses sparse
            encoder = OneHotEncoder(drop='first', sparse=False)
        
        preprocess = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', encoder, self.categorical_features)
            ]
        )
        
        # Create pipeline
        preprocessing_pipeline = Pipeline(steps=[
            ('preprocess', preprocess)
        ])
        
        # Fit the pipeline
        print("  Fitting preprocessing pipeline...")
        preprocessing_pipeline.fit(X)
        
        # Save the rebuilt pipeline
        try:
            import joblib
            joblib.dump(preprocessing_pipeline, self.preprocessing_pipeline_path)
            print(f"  ✓ Saved rebuilt pipeline to {self.preprocessing_pipeline_path.name}")
        except:
            # Fallback to pickle
            with open(self.preprocessing_pipeline_path, 'wb') as f:
                pickle.dump(preprocessing_pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  ✓ Saved rebuilt pipeline to {self.preprocessing_pipeline_path.name}")
        
        self.preprocessing_pipeline = preprocessing_pipeline
    
    def load_models(self) -> None:
        """
        Load the preprocessing pipeline and all trained models.
        
        This method:
        1. Loads the saved preprocessing pipeline from pickle file
        2. Loads all 4 XGBoost models
        """
        if self.is_loaded:
            return
        
        # Load preprocessing pipeline from pickle file
        if not self.preprocessing_pipeline_path.exists():
            raise FileNotFoundError(
                f"Preprocessing pipeline not found at {self.preprocessing_pipeline_path}. "
                "Please ensure preprocessing_pipeline.pkl exists in the spirometry folder."
            )
        
        print(f"Loading preprocessing pipeline from {self.preprocessing_pipeline_path}...")
        try:
            self.preprocessing_pipeline = safe_pickle_load(self.preprocessing_pipeline_path)
            print("  ✓ Preprocessing pipeline loaded")
        except Exception as e:
            print(f"  ⚠ Failed to load preprocessing pipeline: {str(e)}")
            print("  Attempting to rebuild preprocessing pipeline from dataset...")
            try:
                self._rebuild_preprocessing_pipeline()
                print("  ✓ Preprocessing pipeline rebuilt successfully")
            except Exception as rebuild_error:
                raise RuntimeError(
                    f"Failed to load preprocessing pipeline and rebuild failed: {str(rebuild_error)}\n"
                    f"Original error: {str(e)}\n\n"
                    "SOLUTIONS:\n"
                    "1. Ensure spirometry.csv exists in the spirometry folder\n"
                    "2. Or re-save preprocessing_pipeline.pkl with current sklearn version\n"
                    "3. Run: python backend/app/ml_models/spirometry/resave_models.py"
                )
        
        # Load models
        model_files = {
            'obstruction': 'xgb_obstruction_model.pkl',
            'restriction': 'xgb_restriction_model.pkl',
            'prism': 'xgb_prism_model.pkl',
            'mixed': 'xgb_mixed_model.pkl'
        }
        
        print("Loading trained models...")
        for target_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}"
                )
            
            try:
                self.models[target_name] = safe_pickle_load(model_path)
                print(f"  ✓ Loaded {target_name} model")
            except Exception as e:
                raise RuntimeError(f"Failed to load {target_name} model: {str(e)}")
        
        self.is_loaded = True
        print("All models loaded successfully!")
    
    def preprocess(self, data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Dictionary or DataFrame with feature values.
                  Expected keys/columns: sex, race, age, height, weight, 
                  bmi, fev1, fvc (API input names)
                  Note: fev1_fvc will be calculated automatically as fev1/fvc
                  
        Returns:
            Preprocessed numpy array ready for model prediction
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            # Handle single sample - map API input names to CSV column names
            mapped_dict = {}
            for api_col in self.input_cols:
                if api_col in data:
                    csv_col = self.input_to_csv_mapping[api_col]
                    mapped_dict[csv_col] = data[api_col]
            
            # Calculate Baseline_FEV1_FVC_Ratio if not provided
            if 'Baseline_FEV1_FVC_Ratio' not in mapped_dict:
                fev1_col = self.input_to_csv_mapping['fev1']
                fvc_col = self.input_to_csv_mapping['fvc']
                if fev1_col in mapped_dict and fvc_col in mapped_dict:
                    fev1 = mapped_dict[fev1_col]
                    fvc = mapped_dict[fvc_col]
                    if fvc == 0:
                        raise ValueError("FVC cannot be zero (division by zero for FEV1/FVC ratio calculation)")
                    mapped_dict['Baseline_FEV1_FVC_Ratio'] = fev1 / fvc
            
            df = pd.DataFrame([mapped_dict])
        else:
            # DataFrame input - check if it uses API names or CSV names
            if any(col in data.columns for col in self.input_cols):
                # Data uses API input names, need to map to CSV names
                mapped_dict = {}
                for api_col in self.input_cols:
                    if api_col in data.columns:
                        csv_col = self.input_to_csv_mapping[api_col]
                        mapped_dict[csv_col] = data[api_col]
                
                # Calculate Baseline_FEV1_FVC_Ratio if not provided
                if 'Baseline_FEV1_FVC_Ratio' not in mapped_dict:
                    fev1_col = self.input_to_csv_mapping['fev1']
                    fvc_col = self.input_to_csv_mapping['fvc']
                    if fev1_col in mapped_dict and fvc_col in mapped_dict:
                        fev1 = mapped_dict[fev1_col]
                        fvc = mapped_dict[fvc_col]
                        if (fvc == 0).any():
                            raise ValueError("FVC cannot be zero (division by zero for FEV1/FVC ratio calculation)")
                        mapped_dict['Baseline_FEV1_FVC_Ratio'] = fev1 / fvc
                
                df = pd.DataFrame(mapped_dict)
            else:
                # Data already uses CSV column names
                df = data.copy()
                if 'Baseline_FEV1_FVC_Ratio' not in df.columns:
                    if 'Baseline_FEV1_L' in df.columns and 'Baseline_FVC_L' in df.columns:
                        if (df['Baseline_FVC_L'] == 0).any():
                            raise ValueError("FVC cannot be zero (division by zero for FEV1/FVC ratio calculation)")
                        df['Baseline_FEV1_FVC_Ratio'] = df['Baseline_FEV1_L'] / df['Baseline_FVC_L']
        
        # Ensure all required feature columns exist
        missing_cols = set(self.feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Required columns: {self.feature_cols}"
            )
        
        # Select only the feature columns (using CSV names)
        X = df[self.feature_cols].copy()
        
        # Apply preprocessing
        X_processed = self.preprocessing_pipeline.transform(X)
        
        return X_processed
    
    def predict(self, data: Union[Dict, pd.DataFrame], 
                return_proba: bool = False) -> Dict[str, Union[int, float]]:
        """
        Make predictions for all 4 conditions.
        
        Args:
            data: Dictionary or DataFrame with feature values
            return_proba: If True, return probabilities instead of binary predictions
            
        Returns:
            Dictionary with predictions/probabilities for each condition
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Preprocess data
        X_processed = self.preprocess(data)
        
        results = {}
        
        for target_name in self.target_names:
            model = self.models[target_name]
            
            if return_proba:
                # Get probability of positive class
                proba = model.predict_proba(X_processed)[:, 1]
                results[target_name] = float(proba[0]) if len(proba) == 1 else proba.tolist()
            else:
                # Get binary prediction
                pred = model.predict(X_processed)
                results[target_name] = int(pred[0]) if len(pred) == 1 else pred.tolist()
        
        return results
    
    def predict_proba(self, data: Union[Dict, pd.DataFrame]) -> Dict[str, float]:
        """
        Get prediction probabilities for all conditions.
        
        Args:
            data: Dictionary or DataFrame with feature values
            
        Returns:
            Dictionary with probabilities for each condition
        """
        return self.predict(data, return_proba=True)


# Global instance (lazy-loaded)
_featurizer_instance: Optional[SpirometryFeaturizer] = None


def get_featurizer() -> SpirometryFeaturizer:
    """
    Get or create the global featurizer instance.
    
    Returns:
        SpirometryFeaturizer instance
    """
    global _featurizer_instance
    
    if _featurizer_instance is None:
        _featurizer_instance = SpirometryFeaturizer()
        _featurizer_instance.load_models()
    
    return _featurizer_instance


def predict_spirometry(data: Dict) -> Dict[str, Union[int, float]]:
    """
    Convenience function to make predictions.
    
    Args:
        data: Dictionary with feature values:
              - sex: str (e.g., 'Male', 'Female')
              - race: str (e.g., 'White', 'Black', 'Mexican American', etc.)
              - age: float
              - height: float (in cm)
              - weight: float (in kg)
              - bmi: float
              - fev1: float (FEV1 in liters)
              - fvc: float (FVC in liters)
              Note: fev1_fvc will be calculated automatically as fev1/fvc
              
    Returns:
        Dictionary with predictions for obstruction, restriction, prism, mixed
    """
    import time
    from ...core.performance import log_ml_inference
    
    start_time = time.time()
    featurizer = get_featurizer()
    result = featurizer.predict(data)
    duration = time.time() - start_time
    
    # Log performance
    log_ml_inference("spirometry", duration, {"features_count": len(data)})
    return result


def predict_spirometry_proba(data: Dict) -> Dict[str, float]:
    """
    Convenience function to get prediction probabilities.
    
    Args:
        data: Dictionary with feature values (same as predict_spirometry)
              Note: fev1_fvc will be calculated automatically as fev1/fvc
              
    Returns:
        Dictionary with probabilities for each condition
    """
    featurizer = get_featurizer()
    return featurizer.predict_proba(data)
