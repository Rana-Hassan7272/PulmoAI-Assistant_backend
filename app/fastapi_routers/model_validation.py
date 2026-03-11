"""
FastAPI router for model validation and evaluation reports.
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from typing import List, Dict, Optional

router = APIRouter(prefix="/model-validation", tags=["Model Validation"])


@router.get("/reports")
def list_validation_reports():
    """
    List all available model validation reports.
    
    Returns:
        List of available validation reports with metadata
    """
    reports_dir = Path(__file__).parent.parent.parent / "model_validation_reports"
    reports_dir.mkdir(exist_ok=True)
    
    reports = []
    for report_file in sorted(reports_dir.glob("validation_report_*.json"), reverse=True):
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
                reports.append({
                    "filename": report_file.name,
                    "timestamp": data.get("timestamp"),
                    "models_evaluated": data.get("summary", {}).get("models_evaluated", 0),
                    "total_samples": data.get("summary", {}).get("total_samples_tested", 0)
                })
        except Exception:
            continue
    
    return {"reports": reports, "total": len(reports)}


@router.get("/reports/latest")
def get_latest_validation_report():
    """
    Get the most recent model validation report.
    
    Returns:
        Latest validation report with all metrics
    """
    reports_dir = Path(__file__).parent.parent.parent / "model_validation_reports"
    reports_dir.mkdir(exist_ok=True)
    
    report_files = sorted(reports_dir.glob("validation_report_*.json"), reverse=True)
    
    if not report_files:
        raise HTTPException(
            status_code=404,
            detail="No validation reports found. Run evaluation script first: python scripts/evaluate_models.py"
        )
    
    latest_report = report_files[0]
    
    try:
        with open(latest_report, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load validation report: {str(e)}"
        )


@router.get("/reports/{filename}")
def get_validation_report(filename: str):
    """
    Get a specific validation report by filename.
    
    Args:
        filename: Name of the report file
        
    Returns:
        Validation report data
    """
    reports_dir = Path(__file__).parent.parent.parent / "model_validation_reports"
    report_path = reports_dir / filename
    
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Validation report not found: {filename}"
        )
    
    try:
        with open(report_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load validation report: {str(e)}"
        )


@router.get("/summary")
def get_validation_summary():
    """
    Get summary of latest validation results.
    
    Returns:
        Summary with key metrics for each model
    """
    reports_dir = Path(__file__).parent.parent.parent / "model_validation_reports"
    reports_dir.mkdir(exist_ok=True)
    
    report_files = sorted(reports_dir.glob("validation_report_*.json"), reverse=True)
    
    if not report_files:
        return {
            "status": "no_reports",
            "message": "No validation reports found. Run: python scripts/evaluate_models.py"
        }
    
    latest_report = report_files[0]
    
    try:
        with open(latest_report, 'r') as f:
            data = json.load(f)
        
        summary = {
            "report_timestamp": data.get("timestamp"),
            "models": {}
        }
        
        results = data.get("validation_results", {})
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            if model_name == "spirometry":
                summary["models"][model_name] = {
                    "accuracy": model_results.get("overall_accuracy", 0),
                    "precision": model_results.get("overall_precision", 0),
                    "recall": model_results.get("overall_recall", 0),
                    "f1_score": model_results.get("overall_f1_score", 0),
                    "samples_tested": model_results.get("total_samples", 0)
                }
            else:
                summary["models"][model_name] = {
                    "accuracy": model_results.get("accuracy", 0),
                    "precision": model_results.get("precision_macro_avg", 0),
                    "recall": model_results.get("recall_macro_avg", 0),
                    "f1_score": model_results.get("f1_macro_avg", 0),
                    "samples_tested": model_results.get("total_samples", 0)
                }
        
        return summary
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load validation summary: {str(e)}"
        )
