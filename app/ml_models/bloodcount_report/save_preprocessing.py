"""
Script to save preprocessing components (RobustScaler and capping bounds) from training.
Run this after training your model to save the preprocessing components.

Usage:
    After training, save the scaler and bounds:
    
    import joblib
    from sklearn.preprocessing import RobustScaler
    import pandas as pd
    
    # After fitting RobustScaler on training data
    joblib.dump(scaler, 'robust_scaler.pkl')
    
    # After calculating capping bounds
    cap_bounds = {}
    for col in ['PDW', 'NEUTp']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        cap_bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    
    joblib.dump(cap_bounds, 'cap_bounds.pkl')
"""

print("""
To save preprocessing components from your training code, add this after training:

# After fitting RobustScaler
import joblib
joblib.dump(scaler, 'robust_scaler.pkl')

# After calculating capping bounds
cap_bounds = {}
for col in ['PDW', 'NEUTp']:
    Q1 = df_fixed[col].quantile(0.25)
    Q3 = df_fixed[col].quantile(0.75)
    IQR = Q3 - Q1
    cap_bounds[col] = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

joblib.dump(cap_bounds, 'cap_bounds.pkl')

Then place both files in the bloodcount_report folder.
""")

