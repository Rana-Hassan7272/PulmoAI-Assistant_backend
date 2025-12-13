"""
Script to re-save spirometry model files with current Python version.
Run this if you get pickle compatibility errors.
"""

import pickle
from pathlib import Path

# Try importing joblib
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available. Install with: pip install joblib")

def resave_model_file(file_path: Path, output_path: Path = None):
    """
    Re-save a pickle file with current Python version.
    Tries joblib first (for sklearn models), then pickle.
    
    Args:
        file_path: Path to existing pickle file
        output_path: Path to save new file (default: overwrites original)
    """
    if output_path is None:
        output_path = file_path
    
    print(f"Loading {file_path.name}...")
    obj = None
    
    # Try joblib first (sklearn models are often saved with joblib)
    if JOBLIB_AVAILABLE:
        try:
            obj = joblib.load(file_path)
            print(f"  ✓ Loaded with joblib")
        except:
            pass
    
    # Try pickle if joblib didn't work
    if obj is None:
        try:
            with open(file_path, 'rb') as f:
                try:
                    obj = pickle.load(f)
                    print(f"  ✓ Loaded with pickle")
                except:
                    # Try with latin1 encoding
                    f.seek(0)
                    obj = pickle.load(f, encoding='latin1')
                    print(f"  ✓ Loaded with pickle (latin1 encoding)")
        except Exception as e:
            print(f"  ✗ Failed to load: {str(e)}")
            return False
    
    # Re-save with current Python version
    print(f"Re-saving to {output_path.name}...")
    try:
        # Try joblib first (preferred for sklearn models)
        if JOBLIB_AVAILABLE:
            joblib.dump(obj, output_path)
            print(f"  ✓ Saved with joblib")
        else:
            # Fallback to pickle
            with open(output_path, 'wb') as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  ✓ Saved with pickle (protocol {pickle.HIGHEST_PROTOCOL})")
        return True
    except Exception as e:
        print(f"  ✗ Failed to save: {str(e)}")
        return False

if __name__ == "__main__":
    models_dir = Path(__file__).parent
    
    files_to_resave = [
        "preprocessing_pipeline.pkl",
        "xgb_obstruction_model.pkl",
        "xgb_restriction_model.pkl",
        "xgb_prism_model.pkl",
        "xgb_mixed_model.pkl"
    ]
    
    print("Re-saving spirometry model files with current Python version...")
    print("=" * 60)
    
    success_count = 0
    for filename in files_to_resave:
        file_path = models_dir / filename
        if file_path.exists():
            print(f"\nProcessing {filename}...")
            if resave_model_file(file_path):
                success_count += 1
        else:
            print(f"\n{filename} not found, skipping...")
    
    print("\n" + "=" * 60)
    print(f"Successfully re-saved {success_count}/{len(files_to_resave)} files")
    
    if success_count == len(files_to_resave):
        print("\n✓ All files re-saved successfully!")
        print("You can now use the models without compatibility issues.")
    else:
        print("\n⚠ Some files could not be re-saved.")
        print("You may need to re-train the models.")

