# Spirometry ML Models

This module contains trained XGBoost models for predicting spirometry conditions:
- **Obstruction**: Airflow obstruction
- **Restriction**: Restrictive spirometry pattern
- **PRISm**: Preserved Ratio Impaired Spirometry
- **Mixed**: Mixed pattern

## Usage

### Basic Usage

```python
from app.ml_models.spirometry import predict_spirometry, predict_spirometry_proba

# Example input data
patient_data = {
    'sex': 'Male',
    'race': 'White',
    'age': 45.0,
    'height': 175.0,  # cm
    'weight': 80.0,   # kg
    'bmi': 26.1,
    'fev1': 3.2,     # liters
    'fvc': 4.1       # liters
    # fev1_fvc will be calculated automatically as 3.2/4.1 = 0.78
}

# Get binary predictions (0 or 1)
predictions = predict_spirometry(patient_data)
# Returns: {'obstruction': 0, 'restriction': 0, 'prism': 0, 'mixed': 0}

# Get probabilities (0.0 to 1.0)
probabilities = predict_spirometry_proba(patient_data)
# Returns: {'obstruction': 0.15, 'restriction': 0.08, 'prism': 0.12, 'mixed': 0.05}
```

### Advanced Usage

```python
from app.ml_models.spirometry import SpirometryFeaturizer

# Create featurizer instance
featurizer = SpirometryFeaturizer()

# Load models (automatically loads preprocessing pipeline and models)
featurizer.load_models()

# Preprocess data only
processed_data = featurizer.preprocess(patient_data)

# Make predictions
predictions = featurizer.predict(patient_data)
probabilities = featurizer.predict_proba(patient_data)
```

## Input Data Format

Required fields:
- `sex`: str - 'Male' or 'Female'
- `race`: str - Race category (e.g., 'White', 'Black', 'Mexican American', etc.)
- `age`: float - Age in years
- `height`: float - Height in centimeters
- `weight`: float - Weight in kilograms
- `bmi`: float - Body Mass Index
- `fev1`: float - Forced Expiratory Volume in 1 second (liters)
- `fvc`: float - Forced Vital Capacity (liters)

**Note:** `fev1_fvc` ratio is automatically calculated as `fev1 / fvc` - you don't need to provide it!

## Models

The following trained models are included:
- `xgb_obstruction_model.pkl`
- `xgb_restriction_model.pkl`
- `xgb_prism_model.pkl`
- `xgb_mixed_model.pkl`

## Dataset

The training dataset is located at: `dataset/spirometry.csv`

