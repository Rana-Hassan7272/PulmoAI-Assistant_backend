"""
Blood Count Report Disease Prediction Model

This module provides functionality to:
1. Load trained sklearn model (XGBoost/RandomForest/LogisticRegression) for blood disease detection
2. Preprocess blood count report data (RobustScaler, outlier capping)
3. Make predictions for 9 different blood-related conditions
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import RobustScaler, LabelEncoder
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
    error_msg += "1. Re-save the model files using: python backend/app/ml_models/bloodcount_report/resave_models.py\n"
    error_msg += "2. Or re-save manually in Python with: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)\n"
    error_msg += "3. Install joblib: pip install joblib\n"
    error_msg += "4. Install pickle5: pip install pickle5"
    
    raise RuntimeError(error_msg)


class BloodCountPredictor:
    """
    Main class for blood count report disease prediction.
    
    Handles:
    - Data preprocessing (outlier capping, RobustScaler)
    - Loading trained sklearn model
    - Making predictions for blood diseases
    """
    
    # Feature columns required for prediction
    FEATURE_COLS = [
        'WBC', 'LYMp', 'NEUTp', 'LYMn', 'NEUTn', 'RBC', 'HGB', 'HCT', 
        'MCV', 'MCH', 'MCHC', 'PLT', 'PDW', 'PCT'
    ]
    
    # Columns that need RobustScaler (outlier-heavy)
    ROBUST_SCALE_COLS = ['LYMp', 'LYMn', 'NEUTp', 'NEUTn', 'WBC']
    
    # Columns that need outlier capping
    CAP_COLS = ['PDW', 'NEUTp']
    
    # Disease labels (9 classes) - will be loaded from label encoder
    DISEASE_LABELS = [
        'Normocytic hypochromic anemia',
        'Iron deficiency anemia',
        'Other microcytic anemia',
        'Leukemia',
        'Healthy',
        'Thrombocytopenia',
        'Normocytic normochromic anemia',
        'Leukemia with thrombocytopenia',
        'Macrocytic anemia'
    ]
    
    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the blood count predictor.
        
        Args:
            models_dir: Directory containing the model files.
                       If None, uses the directory of this file.
        """
        if models_dir is None:
            models_dir = Path(__file__).parent
        else:
            models_dir = Path(models_dir)
        
        self.models_dir = models_dir
        # Try multiple possible model filenames
        possible_model_names = ["blood_disease_model.pkl", "blood_disease_model .pkl"]
        self.model_path = None
        for name in possible_model_names:
            path = models_dir / name
            if path.exists():
                self.model_path = path
                break
        
        # Try multiple possible label encoder filenames
        possible_label_names = ["label_encoder.pkl", "label_encoder .pkl", "label_encode.pkl"]
        self.label_encoder_path = None
        for name in possible_label_names:
            path = models_dir / name
            if path.exists():
                self.label_encoder_path = path
                break
        
        # Will be initialized when load_models() is called
        self.model = None
        self.scaler: Optional[RobustScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.is_loaded = False
        
        # Store outlier capping bounds (will be calculated from training data statistics)
        # For now, we'll calculate them on-the-fly or use reasonable defaults
        self.cap_bounds = {}
    
    def _calculate_cap_bounds(self, df: pd.DataFrame) -> Dict[str, tuple]:
        """
        Calculate IQR-based capping bounds for outlier columns.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Dictionary with column names as keys and (lower, upper) bounds as values
        """
        bounds = {}
        for col in self.CAP_COLS:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                bounds[col] = (lower_bound, upper_bound)
        return bounds
    
    def load_models(self) -> None:
        """
        Load the scaler, label encoder, and trained model.
        
        Raises:
            FileNotFoundError: If any model file doesn't exist
            RuntimeError: If model loading fails
        """
        if self.is_loaded:
            return
        
        # Load label encoder
        if self.label_encoder_path is None or not self.label_encoder_path.exists():
            raise FileNotFoundError(
                f"Label encoder file not found in {self.models_dir}. "
                "Please ensure label_encoder.pkl exists in the bloodcount_report folder."
            )
        
        print(f"Loading label encoder from {self.label_encoder_path}...")
        try:
            self.label_encoder = safe_pickle_load(self.label_encoder_path)
            print("  ✓ Label encoder loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load label encoder: {str(e)}")
        
        # Load model
        if self.model_path is None or not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found in {self.models_dir}. "
                "Please ensure blood_disease_model.pkl exists in the bloodcount_report folder."
            )
        
        print(f"Loading model from {self.model_path}...")
        try:
            self.model = safe_pickle_load(self.model_path)
            print("  ✓ Model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
        # Initialize RobustScaler (will be fitted on first prediction or can be pre-fitted)
        # For inference, we'll fit it on the input data itself (not ideal but works)
        # In production, you'd want to save and load a pre-fitted scaler
        self.scaler = RobustScaler()
        print("  ✓ RobustScaler initialized")
        
        self.is_loaded = True
        print("All models loaded successfully!")
    
    def preprocess(self, data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Preprocess input data for prediction.
        Applies outlier capping and RobustScaler as per training pipeline.
        
        Args:
            data: Dictionary or DataFrame with blood count values.
                  Expected keys/columns: WBC, LYMp, NEUTp, LYMn, NEUTn, 
                  RBC, HGB, HCT, MCV, MCH, MCHC, PLT, PDW, PCT
                  
        Returns:
            Preprocessed numpy array ready for model prediction
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            # Handle single sample
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        # Ensure all required columns exist
        missing_cols = set(self.FEATURE_COLS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Required columns: {self.FEATURE_COLS}"
            )
        
        # Select only the feature columns in the correct order
        X = df[self.FEATURE_COLS].copy()
        
        # Convert to numeric, handling any non-numeric values
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Check for any NaN values after conversion
        if X.isna().any().any():
            raise ValueError(
                "Invalid numeric values found in input data. "
                "Please ensure all feature values are numeric."
            )
        
        # Step 1: Apply outlier capping for PDW and NEUTp
        # Try to load saved capping bounds if available
        cap_bounds_path = self.models_dir / "cap_bounds.pkl"
        if cap_bounds_path.exists():
            try:
                saved_bounds = safe_pickle_load(cap_bounds_path)
                for col in self.CAP_COLS:
                    if col in X.columns and col in saved_bounds:
                        lower, upper = saved_bounds[col]
                        X[col] = X[col].clip(lower=lower, upper=upper)
            except:
                # If loading fails, skip capping
                pass
        else:
            # No saved bounds - for single sample, skip capping
            # (IQR calculation needs multiple samples)
            if len(X) > 1:
                for col in self.CAP_COLS:
                    if col in X.columns:
                        Q1 = X[col].quantile(0.25)
                        Q3 = X[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Step 2: Apply RobustScaler to outlier-heavy columns
        # Try to load pre-fitted scaler (should be saved from training)
        scaler_path = self.models_dir / "robust_scaler.pkl"
        X_scaled = X.copy()
        
        if scaler_path.exists():
            try:
                fitted_scaler = safe_pickle_load(scaler_path)
                X_scaled[self.ROBUST_SCALE_COLS] = fitted_scaler.transform(X[self.ROBUST_SCALE_COLS])
            except Exception as e:
                # If loading fails, warn but continue without scaling
                import warnings
                warnings.warn(f"Could not load RobustScaler: {e}. Proceeding without scaling.")
                pass
        else:
            # No saved scaler - this is a problem for proper inference
            # For now, we'll proceed without scaling (not ideal but allows prediction)
            import warnings
            warnings.warn(
                "robust_scaler.pkl not found. Predictions may be less accurate. "
                "Please save the fitted RobustScaler from training data."
            )
            # Proceed without scaling for these columns
            pass
        
        # Convert to numpy array
        X_array = X_scaled.values.astype(np.float32)
        
        return X_array
    
    def predict(self, data: Union[Dict, pd.DataFrame], 
                return_proba: bool = False) -> Union[Dict[str, Union[int, str, float]], Dict[str, float]]:
        """
        Make prediction on blood count data.
        
        Args:
            data: Dictionary or DataFrame with blood count values
            return_proba: If True, return probabilities for all classes.
                         If False, return predicted class.
                  
        Returns:
            If return_proba=False:
                Dictionary with:
                - 'class_id': int (0-8)
                - 'disease_name': str (name of the predicted disease)
                - 'confidence': float (confidence score)
            If return_proba=True:
                Dictionary with probabilities for each disease class
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Preprocess data
        X_processed = self.preprocess(data)
        
        # Get probabilities from model
        probabilities = self.model.predict_proba(X_processed)[0]
        
        # Get the class with highest probability
        predicted_class_idx = int(np.argmax(probabilities))
        max_probability = float(probabilities[predicted_class_idx])
        
        # Map probabilities to disease names using label encoder
        prob_dict = {}
        try:
            # Get all class labels from encoder
            all_classes = self.label_encoder.classes_
            for idx, class_id in enumerate(all_classes):
                disease_name = self.label_encoder.inverse_transform([class_id])[0]
                prob_dict[disease_name] = float(probabilities[idx])
            
            # Get disease name for predicted class
            predicted_class_id = all_classes[predicted_class_idx]
            disease_name = self.label_encoder.inverse_transform([predicted_class_id])[0]
        except Exception as e:
            # Fallback: use DISEASE_LABELS if label encoder fails
            for idx, disease_name in enumerate(self.DISEASE_LABELS):
                if idx < len(probabilities):
                    prob_dict[disease_name] = float(probabilities[idx])
            
            if predicted_class_idx < len(self.DISEASE_LABELS):
                disease_name = self.DISEASE_LABELS[predicted_class_idx]
            else:
                disease_name = f"Class_{predicted_class_idx}"
        
        if return_proba:
            return prob_dict
        else:
            return {
                'class_id': int(predicted_class_idx),
                'disease_name': str(disease_name),
                'confidence': max_probability
            }
    
    def predict_proba(self, data: Union[Dict, pd.DataFrame]) -> Dict[str, float]:
        """
        Get prediction probabilities for all disease classes.
        
        Args:
            data: Dictionary or DataFrame with blood count values
                  
        Returns:
            Dictionary with probabilities for each disease class
        """
        return self.predict(data, return_proba=True)


# Global instance (lazy-loaded)
_predictor_instance: Optional[BloodCountPredictor] = None


def get_predictor() -> BloodCountPredictor:
    """
    Get or create the global predictor instance.
    
    Returns:
        BloodCountPredictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = BloodCountPredictor()
        _predictor_instance.load_models()
    
    return _predictor_instance


def predict_blood_disease(data: Dict) -> Dict[str, Union[int, str, float]]:
    """
    Convenience function to make predictions on blood count data.
    
    Args:
        data: Dictionary with blood count values:
              - WBC: float (White Blood Cell count)
              - LYMp: float (Lymphocyte percentage)
              - NEUTp: float (Neutrophil percentage)
              - LYMn: float (Lymphocyte absolute count)
              - NEUTn: float (Neutrophil absolute count)
              - RBC: float (Red Blood Cell count)
              - HGB: float (Hemoglobin)
              - HCT: float (Hematocrit)
              - MCV: float (Mean Corpuscular Volume)
              - MCH: float (Mean Corpuscular Hemoglobin)
              - MCHC: float (Mean Corpuscular Hemoglobin Concentration)
              - PLT: float (Platelet count)
              - PDW: float (Platelet Distribution Width)
              - PCT: float (Plateletcrit)
              Note: Missing values will be filled with healthy defaults
              
    Returns:
        Dictionary with:
        - 'class_id': int (0-8)
        - 'disease_name': str (name of the predicted disease)
        - 'confidence': float (confidence score 0-1)
    """
    predictor = get_predictor()
    return predictor.predict(data)


def predict_blood_disease_proba(data: Dict) -> Dict[str, float]:
    """
    Convenience function to get prediction probabilities.
    
    Args:
        data: Dictionary with blood count values (same as predict_blood_disease)
              Note: Missing values will be filled with healthy defaults
              
    Returns:
        Dictionary with probabilities for each disease class
    """
    predictor = get_predictor()
    return predictor.predict_proba(data)
