# X-Ray Pneumonia Detection ML Models

This module contains a trained ResNet50 model for detecting pneumonia in chest X-ray images.

## Model Details

- **Architecture**: ResNet50 (pretrained, fine-tuned)
- **Classes**: 3 classes
  - `0`: No disease
  - `1`: Bacterial pneumonia
  - `2`: Viral pneumonia
- **Input Size**: 224x224 RGB images
- **Model File**: `pneumonia_resnet50.pth`

## Usage

### Basic Usage

```python
from app.ml_models.xray import predict_xray, predict_xray_proba

# Predict from image file path
result = predict_xray("path/to/xray_image.jpg")
# Returns: {'class_id': 1, 'class_name': 'Bacterial pneumonia', 'confidence': 0.95}

# Get probabilities for all classes
probabilities = predict_xray_proba("path/to/xray_image.jpg")
# Returns: {'No disease': 0.02, 'Bacterial pneumonia': 0.95, 'Viral pneumonia': 0.03}
```

### Using PIL Image

```python
from PIL import Image
from app.ml_models.xray import predict_xray

image = Image.open("xray_image.jpg")
result = predict_xray(image)
```

### Advanced Usage

```python
from app.ml_models.xray import XRayPneumoniaPredictor

# Create predictor instance
predictor = XRayPneumoniaPredictor()

# Load model
predictor.load_model()

# Make prediction
result = predictor.predict("xray_image.jpg")
probabilities = predictor.predict_proba("xray_image.jpg")
```

## Input Format

The model accepts images in multiple formats:
- **File path**: `str` or `Path` object pointing to image file
- **PIL Image**: `PIL.Image.Image` object
- **NumPy array**: `numpy.ndarray` with shape (H, W, C) and values 0-255

Supported image formats: JPEG, PNG, etc. (anything PIL can open)

## Image Preprocessing

Images are automatically:
1. Resized to 224x224 pixels
2. Converted to RGB (if not already)
3. Normalized using ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

## Output Format

### Prediction Result
```python
{
    'class_id': int,        # 0, 1, or 2
    'class_name': str,      # 'No disease', 'Bacterial pneumonia', or 'Viral pneumonia'
    'confidence': float     # Confidence score (0.0 to 1.0)
}
```

### Probability Result
```python
{
    'No disease': float,           # Probability for No disease class
    'Bacterial pneumonia': float,   # Probability for Bacterial pneumonia class
    'Viral pneumonia': float        # Probability for Viral pneumonia class
}
```

## Model Training

The model was trained using:
- ResNet50 architecture (pretrained on ImageNet)
- Weighted CrossEntropyLoss for class imbalance
- Adam optimizer with learning rate 1e-4
- Learning rate scheduler (StepLR)
- Data augmentation (random flips, rotations, color jitter)

