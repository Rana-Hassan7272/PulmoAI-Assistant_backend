"""
ML Model Performance Benchmark Script

Benchmarks inference times for all ML models:
- X-ray pneumonia detection
- Spirometry analysis
- Blood count disease prediction
"""
import sys
import time
import statistics
from pathlib import Path
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Handle NumPy import with error message
try:
    import numpy as np
except ImportError as e:
    print("ERROR: NumPy import failed. This may be due to NumPy version compatibility.")
    print("Try: pip install 'numpy<2.0.0'")
    sys.exit(1)

# Import ML models with error handling
try:
    from app.ml_models.xray.preprocessor import predict_xray, predict_xray_proba
except Exception as e:
    print(f"WARNING: Could not import X-ray model: {e}")
    predict_xray = None
    predict_xray_proba = None

try:
    from app.ml_models.spirometry.featurizer import predict_spirometry, predict_spirometry_proba
except Exception as e:
    print(f"WARNING: Could not import Spirometry model: {e}")
    print("This may be due to NumPy version compatibility. Try: pip install 'numpy<2.0.0'")
    predict_spirometry = None
    predict_spirometry_proba = None

try:
    from app.ml_models.bloodcount_report.feature import predict_blood_disease, predict_blood_disease_proba
except Exception as e:
    print(f"WARNING: Could not import Blood Count model: {e}")
    predict_blood_disease = None
    predict_blood_disease_proba = None


def benchmark_xray_model(num_iterations: int = 10):
    """Benchmark X-ray model inference time."""
    print("\n" + "="*60)
    print("BENCHMARKING X-RAY PNEUMONIA DETECTION MODEL")
    print("="*60)
    
    if predict_xray is None:
        print("SKIPPED: X-ray model not available")
        return None
    
    # Create a dummy test image
    test_image = Image.new('RGB', (224, 224), color='white')
    
    try:
        # Warm-up run
        print("Warming up model...")
        predict_xray(test_image)
        
        # Benchmark prediction
        print(f"Running {num_iterations} inference iterations...")
        times = []
        for i in range(num_iterations):
            start = time.time()
            result = predict_xray(test_image)
            duration = time.time() - start
            times.append(duration)
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_iterations} iterations...")
        
        # Benchmark probability prediction
        print(f"Running {num_iterations} probability prediction iterations...")
        proba_times = []
        for i in range(num_iterations):
            start = time.time()
            result = predict_xray_proba(test_image)
            duration = time.time() - start
            proba_times.append(duration)
        
        # Calculate statistics
        stats = {
            "prediction": {
                "avg": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0,
                "p95": sorted(times)[int(len(times) * 0.95)] if times else 0,
            },
            "probability": {
                "avg": statistics.mean(proba_times),
                "median": statistics.median(proba_times),
                "min": min(proba_times),
                "max": max(proba_times),
                "std": statistics.stdev(proba_times) if len(proba_times) > 1 else 0,
                "p95": sorted(proba_times)[int(len(proba_times) * 0.95)] if proba_times else 0,
            }
        }
        
        print("\nResults:")
        print(f"  Prediction - Avg: {stats['prediction']['avg']:.4f}s, "
              f"Median: {stats['prediction']['median']:.4f}s, "
              f"P95: {stats['prediction']['p95']:.4f}s")
        print(f"  Probability - Avg: {stats['probability']['avg']:.4f}s, "
              f"Median: {stats['probability']['median']:.4f}s, "
              f"P95: {stats['probability']['p95']:.4f}s")
        
        return stats
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def benchmark_spirometry_model(num_iterations: int = 10):
    """Benchmark Spirometry model inference time."""
    print("\n" + "="*60)
    print("BENCHMARKING SPIROMETRY ANALYSIS MODEL")
    print("="*60)
    
    if predict_spirometry is None:
        print("SKIPPED: Spirometry model not available")
        return None
    
    # Test data with all required fields
    test_data = {
        "sex": "Male",
        "race": "White",
        "age": 30,
        "height": 175,
        "weight": 70,
        "bmi": 22.9,
        "fev1": 5.0,
        "fvc": 5.0
    }
    
    try:
        # Warm-up run
        print("Warming up model...")
        predict_spirometry(test_data)
        
        # Benchmark prediction
        print(f"Running {num_iterations} inference iterations...")
        times = []
        for i in range(num_iterations):
            start = time.time()
            result = predict_spirometry(test_data)
            duration = time.time() - start
            times.append(duration)
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_iterations} iterations...")
        
        # Benchmark probability prediction
        print(f"Running {num_iterations} probability prediction iterations...")
        proba_times = []
        for i in range(num_iterations):
            start = time.time()
            result = predict_spirometry_proba(test_data)
            duration = time.time() - start
            proba_times.append(duration)
        
        # Calculate statistics
        stats = {
            "prediction": {
                "avg": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0,
                "p95": sorted(times)[int(len(times) * 0.95)] if times else 0,
            },
            "probability": {
                "avg": statistics.mean(proba_times),
                "median": statistics.median(proba_times),
                "min": min(proba_times),
                "max": max(proba_times),
                "std": statistics.stdev(proba_times) if len(proba_times) > 1 else 0,
                "p95": sorted(proba_times)[int(len(proba_times) * 0.95)] if proba_times else 0,
            }
        }
        
        print("\nResults:")
        print(f"  Prediction - Avg: {stats['prediction']['avg']:.4f}s, "
              f"Median: {stats['prediction']['median']:.4f}s, "
              f"P95: {stats['prediction']['p95']:.4f}s")
        print(f"  Probability - Avg: {stats['probability']['avg']:.4f}s, "
              f"Median: {stats['probability']['median']:.4f}s, "
              f"P95: {stats['probability']['p95']:.4f}s")
        
        return stats
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def benchmark_bloodcount_model(num_iterations: int = 10):
    """Benchmark Blood Count model inference time."""
    print("\n" + "="*60)
    print("BENCHMARKING BLOOD COUNT DISEASE PREDICTION MODEL")
    print("="*60)
    
    if predict_blood_disease is None:
        print("SKIPPED: Blood Count model not available")
        return None
    
    # Test data with all required fields
    test_data = {
        "WBC": 7.0, "LYMp": 35.0, "NEUTp": 60.0, "LYMn": 2.0, "NEUTn": 4.0,
        "RBC": 4.8, "HGB": 14.5, "HCT": 42.0, "MCV": 90.0, "MCH": 30.0,
        "MCHC": 33.0, "PLT": 250.0, "PDW": 12.0, "PCT": 0.25
    }
    
    try:
        # Warm-up run
        print("Warming up model...")
        predict_blood_disease(test_data)
        
        # Benchmark prediction
        print(f"Running {num_iterations} inference iterations...")
        times = []
        for i in range(num_iterations):
            start = time.time()
            result = predict_blood_disease(test_data)
            duration = time.time() - start
            times.append(duration)
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_iterations} iterations...")
        
        # Benchmark probability prediction
        print(f"Running {num_iterations} probability prediction iterations...")
        proba_times = []
        for i in range(num_iterations):
            start = time.time()
            result = predict_blood_disease_proba(test_data)
            duration = time.time() - start
            proba_times.append(duration)
        
        # Calculate statistics
        stats = {
            "prediction": {
                "avg": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "std": statistics.stdev(times) if len(times) > 1 else 0,
                "p95": sorted(times)[int(len(times) * 0.95)] if times else 0,
            },
            "probability": {
                "avg": statistics.mean(proba_times),
                "median": statistics.median(proba_times),
                "min": min(proba_times),
                "max": max(proba_times),
                "std": statistics.stdev(proba_times) if len(proba_times) > 1 else 0,
                "p95": sorted(proba_times)[int(len(proba_times) * 0.95)] if proba_times else 0,
            }
        }
        
        print("\nResults:")
        print(f"  Prediction - Avg: {stats['prediction']['avg']:.4f}s, "
              f"Median: {stats['prediction']['median']:.4f}s, "
              f"P95: {stats['prediction']['p95']:.4f}s")
        print(f"  Probability - Avg: {stats['probability']['avg']:.4f}s, "
              f"Median: {stats['probability']['median']:.4f}s, "
              f"P95: {stats['probability']['p95']:.4f}s")
        
        return stats
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main():
    """Run all ML model benchmarks."""
    print("\n" + "="*60)
    print("ML MODEL PERFORMANCE BENCHMARK SUITE")
    print("="*60)
    print("\nThis script benchmarks inference times for all ML models.")
    print("Each model will be tested with 10 iterations by default.\n")
    
    num_iterations = 10
    if len(sys.argv) > 1:
        try:
            num_iterations = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of iterations: {sys.argv[1]}. Using default: 10")
    
    results = {}
    
    # Benchmark X-ray model
    results["xray"] = benchmark_xray_model(num_iterations)
    
    # Benchmark Spirometry model
    results["spirometry"] = benchmark_spirometry_model(num_iterations)
    
    # Benchmark Blood Count model
    results["bloodcount"] = benchmark_bloodcount_model(num_iterations)
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for model_name, stats in results.items():
        if stats:
            print(f"\n{model_name.upper()}:")
            print(f"  Prediction - Avg: {stats['prediction']['avg']:.4f}s")
            print(f"  Probability - Avg: {stats['probability']['avg']:.4f}s")
    
    print("\n" + "="*60)
    print("Benchmark completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
