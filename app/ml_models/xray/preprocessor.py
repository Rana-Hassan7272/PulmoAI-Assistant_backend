"""
X-Ray Pneumonia Detection Model Preprocessor and Predictor

This module provides functionality to:
1. Load trained ResNet50 model for pneumonia detection
2. Preprocess X-ray images for prediction
3. Make predictions (3 classes: No disease, Bacterial pneumonia, Viral pneumonia)
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
from typing import Union, Dict, Optional, Tuple
from PIL import Image
import numpy as np


class XRayPneumoniaPredictor:
    """
    Main class for X-ray pneumonia detection.
    
    Handles:
    - Image preprocessing (resize, normalize)
    - Loading trained ResNet50 model
    - Making predictions for pneumonia detection
    """
    
    # Class labels (based on class_id: 0, 1, 2)
    CLASS_LABELS = {
        0: 'No disease',
        1: 'Bacterial pneumonia',
        2: 'Viral pneumonia'
    }
    
    # ImageNet normalization (used during training)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the X-ray predictor.
        
        Args:
            model_path: Path to the saved model file (pneumonia_resnet50.pth).
                       If None, uses the default path in the xray folder.
        """
        if model_path is None:
            model_path = Path(__file__).parent / "pneumonia_resnet50.pth"
        else:
            model_path = Path(model_path)
        
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Image preprocessing transform (same as validation transform used in training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
        ])
        
        # Will be initialized when load_model() is called
        self.model: Optional[nn.Module] = None
        self.is_loaded = False
    
    def _build_model(self) -> nn.Module:
        """
        Build the ResNet50 model architecture (same as training).
        
        Returns:
            ResNet50 model with 3 output classes
        """
        model = models.resnet50(pretrained=False)  # We'll load our trained weights
        model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
        return model
    
    def load_model(self) -> None:
        """
        Load the trained model from file.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if self.is_loaded:
            return
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please ensure the trained model is saved in the xray folder."
            )
        
        print(f"Loading model from {self.model_path}...")
        
        # Build model architecture
        self.model = self._build_model()
        
        # Load trained weights
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {str(e)}")
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.is_loaded = True
        print(f"Model loaded successfully on {self.device}!")
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for model prediction.
        
        Args:
            image: Image input - can be:
                  - Path to image file (str or Path)
                  - PIL Image object
                  - numpy array (H, W, C) with values 0-255
                  
        Returns:
            Preprocessed tensor ready for model input (1, 3, 224, 224)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                "Expected str, Path, PIL.Image, or numpy.ndarray"
            )
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict(self, image: Union[str, Path, Image.Image, np.ndarray], 
                return_proba: bool = False) -> Union[Dict[str, Union[int, str, float]], Dict[str, float]]:
        """
        Make prediction on an X-ray image.
        
        Args:
            image: Image input (path, PIL Image, or numpy array)
            return_proba: If True, return probabilities for all classes.
                         If False, return predicted class and confidence.
                  
        Returns:
            If return_proba=False:
                Dictionary with:
                - 'class_id': int (0, 1, or 2)
                - 'class_name': str ('No disease', 'Bacterial pneumonia', or 'Viral pneumonia')
                - 'confidence': float (confidence score for predicted class)
            If return_proba=True:
                Dictionary with probabilities for each class:
                - 'No disease': float
                - 'Bacterial pneumonia': float
                - 'Viral pneumonia': float
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        if return_proba:
            # Return probabilities for all classes
            return {
                'No disease': float(probabilities[0][0].item()),
                'Bacterial pneumonia': float(probabilities[0][1].item()),
                'Viral pneumonia': float(probabilities[0][2].item())
            }
        else:
            # Return predicted class and confidence
            return {
                'class_id': int(predicted_class),
                'class_name': self.CLASS_LABELS[predicted_class],
                'confidence': float(confidence)
            }
    
    def predict_proba(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, float]:
        """
        Get prediction probabilities for all classes.
        
        Args:
            image: Image input (path, PIL Image, or numpy array)
                  
        Returns:
            Dictionary with probabilities for each class:
            - 'No disease': float
            - 'Bacterial pneumonia': float
            - 'Viral pneumonia': float
        """
        return self.predict(image, return_proba=True)


# Global instance (lazy-loaded)
_predictor_instance: Optional[XRayPneumoniaPredictor] = None


def get_predictor() -> XRayPneumoniaPredictor:
    """
    Get or create the global predictor instance.
    
    Returns:
        XRayPneumoniaPredictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = XRayPneumoniaPredictor()
        _predictor_instance.load_model()
    
    return _predictor_instance


def predict_xray(image: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Union[int, str, float]]:
    """
    Convenience function to make predictions on an X-ray image.
    
    Args:
        image: Image input - can be:
              - Path to image file (str or Path)
              - PIL Image object
              - numpy array (H, W, C) with values 0-255
              
    Returns:
        Dictionary with:
        - 'class_id': int (0=No disease, 1=Bacterial pneumonia, 2=Viral pneumonia)
        - 'class_name': str ('No disease', 'Bacterial pneumonia', or 'Viral pneumonia')
        - 'confidence': float (confidence score 0-1)
    """
    import time
    from ...core.performance import log_ml_inference
    
    start_time = time.time()
    predictor = get_predictor()
    result = predictor.predict(image)
    duration = time.time() - start_time
    
    # Log performance
    input_size = None
    if isinstance(image, Image.Image):
        input_size = {"width": image.width, "height": image.height}
    elif isinstance(image, np.ndarray):
        input_size = {"shape": image.shape}
    
    log_ml_inference("xray", duration, input_size)
    return result


def predict_xray_proba(image: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, float]:
    """
    Convenience function to get prediction probabilities.
    
    Args:
        image: Image input (path, PIL Image, or numpy array)
              
    Returns:
        Dictionary with probabilities for each class:
        - 'No disease': float
        - 'Bacterial pneumonia': float
        - 'Viral pneumonia': float
    """
    predictor = get_predictor()
    return predictor.predict_proba(image)
