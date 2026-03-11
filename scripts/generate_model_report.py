"""
Generate Model Validation Report with Confusion Matrices

Creates a comprehensive report with visualizations (if matplotlib available)
or text-based reports.
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_models import (
    evaluate_xray_model,
    evaluate_spirometry_model,
    evaluate_bloodcount_model,
    generate_validation_report
)

# Try to import matplotlib for visualizations
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("NOTE: matplotlib not available. Reports will be text-only.")


def plot_confusion_matrix(cm, labels, title, output_path):
    """Plot confusion matrix if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        return False
    
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"Warning: Could not create confusion matrix plot: {e}")
        return False


def generate_text_report(results: dict, output_path: str):
    """Generate human-readable text report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL VALIDATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write("="*80 + "\n\n")
        
        for model_name, model_results in results.items():
            if not model_results:
                f.write(f"\n{model_name.upper()} MODEL: Evaluation failed or skipped\n")
                f.write("-"*80 + "\n")
                continue
            
            f.write(f"\n{model_name.upper()} MODEL EVALUATION\n")
            f.write("-"*80 + "\n")
            
            if model_name == 'spirometry':
                f.write(f"Total Samples: {model_results.get('total_samples', 0)}\n")
                f.write(f"Overall Accuracy: {model_results.get('overall_accuracy', 0):.4f} ({model_results.get('overall_accuracy', 0)*100:.2f}%)\n")
                f.write(f"Overall Precision: {model_results.get('overall_precision', 0):.4f}\n")
                f.write(f"Overall Recall: {model_results.get('overall_recall', 0):.4f}\n")
                f.write(f"Overall F1 Score: {model_results.get('overall_f1_score', 0):.4f}\n\n")
                
                f.write("Per-Condition Metrics:\n")
                for condition, metrics in model_results.get('per_condition_metrics', {}).items():
                    f.write(f"  {condition}:\n")
                    f.write(f"    Accuracy: {metrics.get('accuracy', 0):.4f}\n")
                    f.write(f"    Precision: {metrics.get('precision', 0):.4f}\n")
                    f.write(f"    Recall: {metrics.get('recall', 0):.4f}\n")
                    f.write(f"    F1 Score: {metrics.get('f1_score', 0):.4f}\n")
            else:
                f.write(f"Total Samples: {model_results.get('total_samples', 0)}\n")
                f.write(f"Accuracy: {model_results.get('accuracy', 0):.4f} ({model_results.get('accuracy', 0)*100:.2f}%)\n")
                f.write(f"Precision (macro avg): {model_results.get('precision_macro_avg', 0):.4f}\n")
                f.write(f"Recall (macro avg): {model_results.get('recall_macro_avg', 0):.4f}\n")
                f.write(f"F1 Score (macro avg): {model_results.get('f1_macro_avg', 0):.4f}\n\n")
                
                # Confusion Matrix
                cm = model_results.get('confusion_matrix', [])
                if cm:
                    f.write("Confusion Matrix:\n")
                    labels = model_results.get('class_labels', [])
                    if labels:
                        f.write(f"  {'True\\Pred':>15}")
                        for label in labels[:5]:  # Show first 5
                            f.write(f"{str(label)[:10]:>12}")
                        f.write("\n")
                        for i, label in enumerate(labels[:5]):
                            f.write(f"  {str(label)[:15]:>15}")
                            for j in range(min(5, len(labels))):
                                if i < len(cm) and j < len(cm[i]):
                                    f.write(f"{cm[i][j]:>12}")
                                else:
                                    f.write(f"{0:>12}")
                            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Text report saved to: {output_path}")
    return str(output_path)


def main():
    """Generate comprehensive model validation report."""
    print("\n" + "="*70)
    print("GENERATING MODEL VALIDATION REPORT")
    print("="*70 + "\n")
    
    # Run evaluations
    results = {
        'xray': evaluate_xray_model(),
        'spirometry': evaluate_spirometry_model(),
        'bloodcount': evaluate_bloodcount_model()
    }
    
    # Generate JSON report
    json_report_path = generate_validation_report(results)
    
    # Generate text report
    reports_dir = Path(__file__).parent.parent / "model_validation_reports"
    text_report_path = reports_dir / f"validation_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
    generate_text_report(results, text_report_path)
    
    # Generate confusion matrix plots if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating confusion matrix visualizations...")
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            cm = model_results.get('confusion_matrix', [])
            labels = model_results.get('class_labels', [])
            
            if cm and labels:
                plot_path = reports_dir / f"{model_name}_confusion_matrix.png"
                if plot_confusion_matrix(cm, labels, f"{model_name.upper()} Confusion Matrix", plot_path):
                    print(f"  Saved: {plot_path}")
    
    print("\n" + "="*70)
    print("All reports generated successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
