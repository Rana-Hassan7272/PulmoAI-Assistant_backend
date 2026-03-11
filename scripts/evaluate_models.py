"""
Model Validation and Evaluation Scripts

Generates accuracy reports, confusion matrices, and classification metrics for:
- X-ray pneumonia detection model
- Spirometry analysis model
- Blood count disease prediction model
"""
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available. Some metrics will be calculated manually.")


def calculate_metrics_manual(y_true: List, y_pred: List, labels: List) -> Dict:
    """Calculate classification metrics manually if sklearn not available."""
    metrics = {}
    
    # Accuracy
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    metrics['accuracy'] = correct / len(y_true) if y_true else 0.0
    
    # Per-class metrics
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[f'{label}_precision'] = precision
        metrics[f'{label}_recall'] = recall
        metrics[f'{label}_f1'] = f1
    
    # Macro averages
    precisions = [metrics.get(f'{label}_precision', 0) for label in labels]
    recalls = [metrics.get(f'{label}_recall', 0) for label in labels]
    f1_scores = [metrics.get(f'{label}_f1', 0) for label in labels]
    
    metrics['macro_avg_precision'] = sum(precisions) / len(precisions) if precisions else 0.0
    metrics['macro_avg_recall'] = sum(recalls) / len(recalls) if recalls else 0.0
    metrics['macro_avg_f1'] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return metrics


def generate_confusion_matrix_manual(y_true: List, y_pred: List, labels: List) -> List[List[int]]:
    """Generate confusion matrix manually."""
    cm = [[0 for _ in labels] for _ in labels]
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_idx.get(true, 0)
        pred_idx = label_to_idx.get(pred, 0)
        cm[true_idx][pred_idx] += 1
    
    return cm


def evaluate_xray_model(test_data: Optional[List[Tuple]] = None) -> Dict:
    """
    Evaluate X-ray pneumonia detection model.
    
    Args:
        test_data: List of (image_path, true_label) tuples. If None, tries to load from test folder.
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*70)
    print("EVALUATING X-RAY PNEUMONIA DETECTION MODEL")
    print("="*70)
    
    try:
        from app.ml_models.xray.preprocessor import predict_xray, XRayPneumoniaPredictor
        from PIL import Image
    except ImportError as e:
        print(f"ERROR: Could not import X-ray model: {e}")
        return None
    
    # If no test data provided, try to load from test folder
    if test_data is None:
        test_dir = Path(__file__).parent.parent / "app" / "ml_models" / "xray" / "test"
        normal_dir = test_dir / "NORMAL"
        pneumonia_dir = test_dir / "PNEUMONIA"
        
        # Check if test folders exist
        if normal_dir.exists() and pneumonia_dir.exists():
            print(f"Found test dataset in: {test_dir}")
            test_data = []
            
            # Load NORMAL images (label: 0 = No disease)
            normal_images = list(normal_dir.glob("*.jpeg")) + list(normal_dir.glob("*.jpg")) + list(normal_dir.glob("*.png"))
            print(f"  Found {len(normal_images)} NORMAL images")
            for img_path in normal_images:
                test_data.append((img_path, 0))
            
            # Load PNEUMONIA images and classify by filename
            pneumonia_images = list(pneumonia_dir.glob("*.jpeg")) + list(pneumonia_dir.glob("*.jpg")) + list(pneumonia_dir.glob("*.png"))
            print(f"  Found {len(pneumonia_images)} PNEUMONIA images")
            
            viral_count = 0
            bacterial_count = 0
            
            for img_path in pneumonia_images:
                filename_lower = img_path.name.lower()
                # Check for viral pneumonia indicators
                if '_virus_' in filename_lower or 'virus' in filename_lower:
                    test_data.append((img_path, 2))  # Viral pneumonia
                    viral_count += 1
                # Check for bacterial pneumonia indicators
                elif '_bacteri_a_' in filename_lower or 'bacteria' in filename_lower or '_bacteri' in filename_lower:
                    test_data.append((img_path, 1))  # Bacterial pneumonia
                    bacterial_count += 1
                else:
                    # Default to bacterial if unclear (most common)
                    test_data.append((img_path, 1))
                    bacterial_count += 1
            
            print(f"    - Viral pneumonia: {viral_count}")
            print(f"    - Bacterial pneumonia: {bacterial_count}")
            print(f"\nTotal test samples: {len(test_data)}")
            
            # Limit to reasonable size for evaluation (if too large)
            if len(test_data) > 500:
                import random
                random.seed(42)
                test_data = random.sample(test_data, 500)
                print(f"Using random sample of 500 for evaluation")
        else:
            print("NOTE: No test dataset found in test folder.")
            print("OPTION 1: Download test set from Kaggle (Chest X-Ray Pneumonia dataset)")
            print("OPTION 2: Place test images in: backend/app/ml_models/xray/test/")
            print("          - NORMAL/ folder for normal X-rays")
            print("          - PNEUMONIA/ folder for pneumonia X-rays (viral/bacterial)")
            print("\nRunning model validation mode (no accuracy metrics without test data)...")
            
            # Validate model can load and make predictions
            try:
                predictor = XRayPneumoniaPredictor()
                predictor.load_model()
                print("✓ Model loaded successfully")
                
                # Create a single test image to verify prediction works
                test_img = Image.new('RGB', (224, 224), color='gray')
                result = predict_xray(test_img)
                print(f"✓ Model can make predictions: {result.get('class_name', 'Unknown')}")
                
                # Return validation result (not accuracy metrics)
                return {
                    'model': 'xray_pneumonia_detection',
                    'status': 'model_validated',
                    'message': 'Model loads and makes predictions. Test dataset needed for accuracy metrics.',
                    'total_samples': 0,
                    'accuracy': None,
                    'precision_macro_avg': None,
                    'recall_macro_avg': None,
                    'f1_macro_avg': None
                }
            except Exception as e:
                print(f"ERROR: Model validation failed: {e}")
                return None
    
    y_true = []
    y_pred = []
    
    print(f"Evaluating on {len(test_data)} test samples...")
    
    for idx, (image_input, true_label) in enumerate(test_data):
        try:
            # Handle image input - can be path (str/Path) or PIL Image
            if isinstance(image_input, (str, Path)):
                # Load image from path
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                image = image_input
            
            # Make prediction
            result = predict_xray(image)
            predicted_class = result.get('class_id', result.get('class_name', 0))
            
            # Handle class_name if class_id not available
            if isinstance(predicted_class, str):
                class_map = {'No disease': 0, 'Bacterial pneumonia': 1, 'Viral pneumonia': 2}
                predicted_class = class_map.get(predicted_class, 0)
            
            y_true.append(true_label)
            y_pred.append(predicted_class)
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(test_data)} samples...")
                
        except Exception as e:
            print(f"  Warning: Failed to process sample {idx}: {e}")
            continue
    
    if not y_true:
        print("ERROR: No valid predictions made.")
        return None
    
    # Calculate metrics
    labels = [0, 1, 2]  # No disease, Bacterial, Viral
    label_names = ['No disease', 'Bacterial pneumonia', 'Viral pneumonia']
    
    if SKLEARN_AVAILABLE:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, label_name in enumerate(label_names):
            per_class_metrics[label_name] = {
                'precision': precision_score(y_true, y_pred, labels=[i], average='macro', zero_division=0),
                'recall': recall_score(y_true, y_pred, labels=[i], average='macro', zero_division=0),
                'f1': f1_score(y_true, y_pred, labels=[i], average='macro', zero_division=0)
            }
    else:
        metrics = calculate_metrics_manual(y_true, y_pred, labels)
        accuracy = metrics['accuracy']
        precision = metrics['macro_avg_precision']
        recall = metrics['macro_avg_recall']
        f1 = metrics['macro_avg_f1']
        cm = generate_confusion_matrix_manual(y_true, y_pred, labels)
        
        per_class_metrics = {}
        for i, label_name in enumerate(label_names):
            per_class_metrics[label_name] = {
                'precision': metrics.get(f'{i}_precision', 0),
                'recall': metrics.get(f'{i}_recall', 0),
                'f1': metrics.get(f'{i}_f1', 0)
            }
    
    results = {
        'model': 'xray_pneumonia_detection',
        'total_samples': len(y_true),
        'accuracy': float(accuracy),
        'precision_macro_avg': float(precision),
        'recall_macro_avg': float(recall),
        'f1_macro_avg': float(f1),
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm,
        'class_labels': label_names
    }
    
    print("\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision (macro avg): {precision:.4f}")
    print(f"  Recall (macro avg): {recall:.4f}")
    print(f"  F1 Score (macro avg): {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {'':15} {'Predicted:':>20}")
    print(f"  {'':15} {'No disease':>15} {'Bacterial':>15} {'Viral':>15}")
    for i, label_name in enumerate(label_names):
        row = f"  {label_name[:15]:15}"
        for j in range(3):
            row += f"{cm[i][j]:>15}"
        print(row)
    
    return results


def evaluate_spirometry_model(test_data_path: Optional[str] = None) -> Dict:
    """
    Evaluate Spirometry analysis model.
    
    Args:
        test_data_path: Path to CSV file with test data. If None, uses available dataset.
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*70)
    print("EVALUATING SPIROMETRY ANALYSIS MODEL")
    print("="*70)
    
    try:
        from app.ml_models.spirometry.featurizer import predict_spirometry
        import pandas as pd
    except ImportError as e:
        print(f"ERROR: Could not import Spirometry model: {e}")
        return None
    
    # Try to load test data
    found_targets = {}
    df = None
    
    if test_data_path is None:
        # Try to find dataset in spirometry directory
        spirometry_dir = Path(__file__).parent.parent / "app" / "ml_models" / "spirometry"
        possible_paths = [
            spirometry_dir / "dataset" / "spirometry.csv",
            spirometry_dir / "spirometry.csv"
        ]
        
        for path in possible_paths:
            if path.exists():
                test_data_path = str(path)
                break
    
    if test_data_path and Path(test_data_path).exists():
        print(f"Loading test data from: {test_data_path}")
        try:
            df = pd.read_csv(test_data_path)
            print(f"Loaded {len(df)} samples from dataset")
            
            # Use a subset for evaluation (if dataset is large)
            if len(df) > 200:
                df = df.sample(n=200, random_state=42)
                print(f"Using random sample of 200 for evaluation")
            
            # Check for target columns - try different naming conventions
            # The dataset uses columns like: Obstruction_5th_Segmented, Restrictive_Spirometry_Pattern_5th_Segmented, etc.
            target_mapping = {
                'obstruction': ['Obstruction_5th_Segmented', 'Obstruction_5th_GAMLSS'],
                'restriction': ['Restrictive_Spirometry_Pattern_5th_Segmented', 'Restrictive_Spirometry_Pattern_5th_GAMLSS'],
                'prism': ['PRISm_Segmented_5th', 'PRISm_GAMLSS_5th'],
                'mixed': ['Mixed_5th_Segmented', 'Mixed_5th_GAMLSS']
            }
            
            # Find which target columns exist
            for target_name, possible_cols in target_mapping.items():
                for col in possible_cols:
                    if col in df.columns:
                        found_targets[target_name] = col
                        break
            
            if len(found_targets) == 0:
                # Try lowercase versions
                for target_name, possible_cols in target_mapping.items():
                    for col in possible_cols:
                        matching_cols = [c for c in df.columns if c.lower() == col.lower()]
                        if matching_cols:
                            found_targets[target_name] = matching_cols[0]
                            break
            
            if len(found_targets) > 0:
                print(f"Found target columns: {list(found_targets.keys())}")
            else:
                print("NOTE: Dataset doesn't have recognizable target columns. Generating synthetic validation.")
                df = None
        except Exception as e:
            print(f"WARNING: Could not load dataset: {e}")
            df = None
    
    if df is None:
        print("NOTE: No test dataset available. Generating synthetic validation report.")
        print("For accurate metrics, provide test dataset CSV with target columns.")
        
        # Create synthetic test cases
        synthetic_data = []
        for i in range(20):
            synthetic_data.append({
                'sex': 'Male' if i % 2 == 0 else 'Female',
                'race': 'White',
                'age': 30 + (i % 40),
                'height': 170 + (i % 20),
                'weight': 70 + (i % 30),
                'bmi': 22 + (i % 5),
                'fev1': 3.5 + (i % 2),
                'fvc': 4.5 + (i % 2),
                'obstruction': i % 2,
                'restriction': (i + 1) % 2,
                'prism': (i + 2) % 2,
                'mixed': (i + 3) % 2
            })
        df = pd.DataFrame(synthetic_data)
        found_targets = {'obstruction': 'obstruction', 'restriction': 'restriction', 
                        'prism': 'prism', 'mixed': 'mixed'}
        print(f"Generated {len(df)} synthetic test cases")
    
    # Evaluate each target
    all_results = {}
    
    print(f"\nEvaluating on {len(df)} samples with target columns: {list(found_targets.keys())}")
    
    for target_name, target_col in found_targets.items():
        print(f"\nEvaluating {target_name} prediction...")
        
        if target_col not in df.columns:
            print(f"  WARNING: Target column '{target_col}' not found. Skipping.")
            continue
        
        # Convert target column values to binary (yes/no -> 1/0, or keep as is if already numeric)
        y_true_raw = df[target_col].tolist()
        y_true = []
        for val in y_true_raw:
            if pd.isna(val):
                y_true.append(0)
            elif isinstance(val, str):
                # Convert yes/no/No to 1/0
                val_lower = val.lower().strip()
                y_true.append(1 if val_lower in ['yes', '1', 'true'] else 0)
            else:
                y_true.append(int(val) if val else 0)
        y_pred = []
        
        # Make predictions
        for idx, row in df.iterrows():
            try:
                # Prepare input data - map CSV columns to API input format
                # CSV has: Sex, Race, Age, Height, Weight, BMI, Baseline_FEV1_L, Baseline_FVC_L
                # API expects: sex, race, age, height, weight, bmi, fev1, fvc
                input_data = {
                    'sex': row.get('Sex', row.get('sex', 'Male')),
                    'race': row.get('Race', row.get('race', 'White')),
                    'age': row.get('Age', row.get('age', 45)),
                    'height': row.get('Height', row.get('height', 170)),
                    'weight': row.get('Weight', row.get('weight', 70)),
                    'bmi': row.get('BMI', row.get('bmi', 24)),
                    'fev1': row.get('Baseline_FEV1_L', row.get('fev1', 3.5)),
                    'fvc': row.get('Baseline_FVC_L', row.get('fvc', 4.5))
                }
                
                result = predict_spirometry(input_data)
                predicted = result.get(target_name, 0)
                y_pred.append(int(predicted))
                
            except Exception as e:
                print(f"  Warning: Failed to predict for sample {idx}: {e}")
                y_pred.append(0)  # Default prediction
                continue
        
        if not y_true or not y_pred:
            continue
        
        # Calculate metrics
        if SKLEARN_AVAILABLE:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        else:
            metrics = calculate_metrics_manual(y_true, y_pred, [0, 1])
            accuracy = metrics['accuracy']
            precision = metrics.get('1_precision', 0)
            recall = metrics.get('1_recall', 0)
            f1 = metrics.get('1_f1', 0)
            cm = generate_confusion_matrix_manual(y_true, y_pred, [0, 1])
        
        all_results[target_name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm,
            'total_samples': len(y_true)
        }
        
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    
    # Overall results
    if all_results:
        avg_accuracy = sum(r['accuracy'] for r in all_results.values()) / len(all_results)
        avg_precision = sum(r['precision'] for r in all_results.values()) / len(all_results)
        avg_recall = sum(r['recall'] for r in all_results.values()) / len(all_results)
        avg_f1 = sum(r['f1_score'] for r in all_results.values()) / len(all_results)
        
        results = {
            'model': 'spirometry_analysis',
            'overall_accuracy': float(avg_accuracy),
            'overall_precision': float(avg_precision),
            'overall_recall': float(avg_recall),
            'overall_f1_score': float(avg_f1),
            'per_condition_metrics': all_results,
            'total_samples': all_results[list(all_results.keys())[0]]['total_samples'] if all_results else 0
        }
        
        print("\n" + "-"*70)
        print("OVERALL RESULTS:")
        print(f"  Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average F1 Score: {avg_f1:.4f}")
        
        return results
    
    return None


def evaluate_bloodcount_model(test_data_path: Optional[str] = None) -> Dict:
    """
    Evaluate Blood Count disease prediction model.
    
    Args:
        test_data_path: Path to CSV file with test data. If None, uses synthetic validation.
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*70)
    print("EVALUATING BLOOD COUNT DISEASE PREDICTION MODEL")
    print("="*70)
    
    try:
        from app.ml_models.bloodcount_report.feature import predict_blood_disease, BloodCountPredictor
        import pandas as pd
    except ImportError as e:
        print(f"ERROR: Could not import Blood Count model: {e}")
        return None
    
    # Load predictor to access label encoder
    predictor = BloodCountPredictor()
    try:
        predictor.load_models()
        label_encoder = predictor.label_encoder
        disease_labels = predictor.DISEASE_LABELS if hasattr(predictor, 'DISEASE_LABELS') else []
    except Exception as e:
        print(f"WARNING: Could not load models for label mapping: {e}")
        label_encoder = None
        disease_labels = []
    
    # Try to load test data
    df = None
    if test_data_path is None:
        # Try to find dataset in bloodcount directory
        bloodcount_dir = Path(__file__).parent.parent / "app" / "ml_models" / "bloodcount_report"
        possible_paths = [
            bloodcount_dir / "dataset.csv.csv",
            bloodcount_dir / "dataset.csv",
            bloodcount_dir / "dataset" / "dataset.csv"
        ]
        
        for path in possible_paths:
            if path.exists():
                test_data_path = str(path)
                break
    
    if test_data_path and Path(test_data_path).exists():
        print(f"Loading test data from: {test_data_path}")
        try:
            df = pd.read_csv(test_data_path)
            print(f"Loaded {len(df)} samples from dataset")
            
            # Use a subset for evaluation (if dataset is large)
            if len(df) > 200:
                df = df.sample(n=200, random_state=42)
                print(f"Using random sample of 200 for evaluation")
            
            # Check if dataset has Diagnosis column
            if 'Diagnosis' not in df.columns:
                print("WARNING: Dataset doesn't have 'Diagnosis' column. Generating synthetic validation.")
                df = None
        except Exception as e:
            print(f"WARNING: Could not load dataset: {e}")
            df = None
    
    if df is None:
        print("NOTE: No test dataset available. Generating synthetic validation report.")
        print("For accurate metrics, provide test dataset CSV with 'Diagnosis' column.")
        
        # Create synthetic test cases
        synthetic_data = []
        for i in range(30):
            synthetic_data.append({
                'WBC': 7.0 + (i % 3),
                'LYMp': 35.0 + (i % 5),
                'NEUTp': 60.0 + (i % 10),
                'LYMn': 2.0 + (i % 1),
                'NEUTn': 4.0 + (i % 2),
                'RBC': 4.8 + (i % 0.5),
                'HGB': 14.5 + (i % 2),
                'HCT': 42.0 + (i % 5),
                'MCV': 90.0 + (i % 5),
                'MCH': 30.0 + (i % 2),
                'MCHC': 33.0 + (i % 1),
                'PLT': 250.0 + (i % 50),
                'PDW': 12.0 + (i % 2),
                'PCT': 0.25 + (i % 0.1),
                'Diagnosis': ['Normocytic hypochromic anemia', 'Iron deficiency anemia', 'Other microcytic anemia',
                             'Leukemia', 'Healthy', 'Thrombocytopenia', 'Normocytic normochromic anemia',
                             'Leukemia with thrombocytopenia', 'Macrocytic anemia'][i % 9]
            })
        df = pd.DataFrame(synthetic_data)
        print(f"Generated {len(df)} synthetic test cases")
    
    y_true = []
    y_pred = []
    y_pred_names = []
    
    print(f"Evaluating on {len(df)} test samples...")
    
    for idx, row in df.iterrows():
        try:
            # Prepare input data
            input_data = {
                'WBC': row['WBC'], 'LYMp': row['LYMp'], 'NEUTp': row['NEUTp'],
                'LYMn': row['LYMn'], 'NEUTn': row['NEUTn'], 'RBC': row['RBC'],
                'HGB': row['HGB'], 'HCT': row['HCT'], 'MCV': row['MCV'],
                'MCH': row['MCH'], 'MCHC': row['MCHC'], 'PLT': row['PLT'],
                'PDW': row['PDW'], 'PCT': row['PCT']
            }
            
            result = predict_blood_disease(input_data)
            predicted_class_id = result.get('class_id', 0)
            predicted_name = result.get('disease_name', 'Unknown')
            
            # Get true label - either from Diagnosis column (disease name) or numeric disease column
            if 'Diagnosis' in row:
                disease_name = str(row['Diagnosis']).strip()
                try:
                    # Try label encoder first (most accurate)
                    if label_encoder:
                        true_class_id = label_encoder.transform([disease_name])[0]
                    # Fallback to disease labels list
                    elif disease_name in disease_labels:
                        true_class_id = disease_labels.index(disease_name)
                    else:
                        # Last resort: use hash mod 9
                        true_class_id = hash(disease_name) % 9
                except Exception as e:
                    # If label encoder fails, try disease labels
                    if disease_name in disease_labels:
                        true_class_id = disease_labels.index(disease_name)
                    else:
                        true_class_id = hash(disease_name) % 9
            elif 'disease' in row:
                true_class_id = int(row['disease'])
            else:
                true_class_id = 0  # Default
            
            y_true.append(int(true_class_id))
            y_pred.append(int(predicted_class_id))
            y_pred_names.append(predicted_name)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(df)} samples...")
                
        except Exception as e:
            print(f"  Warning: Failed to predict for sample {idx}: {e}")
            y_pred.append(0)
            continue
    
    if not y_true:
        print("ERROR: No valid predictions made.")
        return None
    
    # Calculate metrics
    unique_labels = sorted(list(set(y_true + y_pred)))
    
    if SKLEARN_AVAILABLE:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    else:
        metrics = calculate_metrics_manual(y_true, y_pred, unique_labels)
        accuracy = metrics['accuracy']
        precision = metrics['macro_avg_precision']
        recall = metrics['macro_avg_recall']
        f1 = metrics['macro_avg_f1']
        cm = generate_confusion_matrix_manual(y_true, y_pred, unique_labels)
    
    results = {
        'model': 'bloodcount_disease_prediction',
        'total_samples': len(y_true),
        'accuracy': float(accuracy),
        'precision_macro_avg': float(precision),
        'recall_macro_avg': float(recall),
        'f1_macro_avg': float(f1),
        'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm,
        'class_labels': unique_labels
    }
    
    print("\nResults:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision (macro avg): {precision:.4f}")
    print(f"  Recall (macro avg): {recall:.4f}")
    print(f"  F1 Score (macro avg): {f1:.4f}")
    print(f"\nConfusion Matrix (showing first 5x5 for readability):")
    print(f"  {'True\\Pred':>10}", end="")
    for j in range(min(5, len(unique_labels))):
        print(f"{unique_labels[j]:>8}", end="")
    print()
    for i in range(min(5, len(unique_labels))):
        print(f"  {unique_labels[i]:>10}", end="")
        for j in range(min(5, len(unique_labels))):
            print(f"{cm[i][j] if i < len(cm) and j < len(cm[i]) else 0:>8}", end="")
        print()
    
    return results


def generate_validation_report(all_results: Dict, output_path: Optional[str] = None) -> str:
    """
    Generate comprehensive validation report.
    
    Args:
        all_results: Dictionary with results from all model evaluations
        output_path: Optional path to save report
        
    Returns:
        Path to saved report
    """
    if output_path is None:
        reports_dir = Path(__file__).parent.parent / "model_validation_reports"
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / f"validation_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'validation_results': all_results,
        'summary': {
            'models_evaluated': len([r for r in all_results.values() if r is not None]),
            'total_samples_tested': sum(
                r.get('total_samples', 0) 
                for r in all_results.values() 
                if r and isinstance(r, dict)
            )
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*70)
    print(f"VALIDATION REPORT SAVED: {output_path}")
    print("="*70)
    
    return str(output_path)


def main():
    """Run all model evaluations and generate report."""
    print("\n" + "="*70)
    print("MODEL VALIDATION AND EVALUATION SUITE")
    print("="*70)
    print("\nThis script evaluates ML models and generates accuracy reports.")
    print("NOTE: If test datasets are not available, synthetic validation is performed.\n")
    
    all_results = {}
    
    # Evaluate X-ray model
    all_results['xray'] = evaluate_xray_model()
    
    # Evaluate Spirometry model
    all_results['spirometry'] = evaluate_spirometry_model()
    
    # Evaluate Blood Count model
    all_results['bloodcount'] = evaluate_bloodcount_model()
    
    # Generate comprehensive report
    report_path = generate_validation_report(all_results)
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for model_name, results in all_results.items():
        if results:
            if model_name == 'spirometry':
                acc = results.get('overall_accuracy', 0)
            else:
                acc = results.get('accuracy', None)
            
            if acc is not None:
                print(f"{model_name.upper()}: Accuracy = {acc:.4f} ({acc*100:.2f}%)")
            else:
                status = results.get('status', 'validated')
                message = results.get('message', 'Model validated')
                print(f"{model_name.upper()}: {status.upper()} - {message}")
        else:
            print(f"{model_name.upper()}: Evaluation failed or skipped")
    
    print("\n" + "="*70)
    print("Validation completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
