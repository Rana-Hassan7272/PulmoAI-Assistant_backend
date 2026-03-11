"""
Run All Performance Benchmarks

Executes all benchmark scripts and generates a comprehensive report.
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Import benchmark functions
from benchmark_ml_models import (
    benchmark_xray_model,
    benchmark_spirometry_model,
    benchmark_bloodcount_model
)
from benchmark_rag import benchmark_rag_retrieval
from benchmark_database import benchmark_select_queries, benchmark_insert_queries


def main():
    """Run all benchmarks and generate report."""
    print("\n" + "="*70)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK SUITE")
    print("="*70)
    print("\nThis script runs all performance benchmarks and generates a report.\n")
    
    num_iterations = 10
    if len(sys.argv) > 1:
        try:
            num_iterations = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of iterations: {sys.argv[1]}. Using default: 10")
    
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iterations": num_iterations,
        "benchmarks": {}
    }
    
    # Run ML model benchmarks
    print("\n" + "="*70)
    print("RUNNING ML MODEL BENCHMARKS")
    print("="*70)
    results["benchmarks"]["ml_models"] = {
        "xray": benchmark_xray_model(num_iterations),
        "spirometry": benchmark_spirometry_model(num_iterations),
        "bloodcount": benchmark_bloodcount_model(num_iterations),
    }
    
    # Run RAG benchmark
    print("\n" + "="*70)
    print("RUNNING RAG BENCHMARKS")
    print("="*70)
    results["benchmarks"]["rag"] = benchmark_rag_retrieval(num_iterations)
    
    # Run Database benchmarks
    print("\n" + "="*70)
    print("RUNNING DATABASE BENCHMARKS")
    print("="*70)
    results["benchmarks"]["database"] = {
        "select": benchmark_select_queries(num_iterations * 10),
        "insert": benchmark_insert_queries(num_iterations * 5),
    }
    
    # Save report
    report_dir = Path(__file__).parent.parent / "performance_reports"
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / f"benchmark_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    # Print summary
    if results["benchmarks"]["ml_models"]["xray"]:
        xray_avg = results["benchmarks"]["ml_models"]["xray"]["prediction"]["avg"]
        print(f"\nX-ray Model: {xray_avg:.4f}s avg inference")
    
    if results["benchmarks"]["ml_models"]["spirometry"]:
        spiro_avg = results["benchmarks"]["ml_models"]["spirometry"]["prediction"]["avg"]
        print(f"Spirometry Model: {spiro_avg:.4f}s avg inference")
    
    if results["benchmarks"]["ml_models"]["bloodcount"]:
        blood_avg = results["benchmarks"]["ml_models"]["bloodcount"]["prediction"]["avg"]
        print(f"Blood Count Model: {blood_avg:.4f}s avg inference")
    
    if results["benchmarks"]["rag"]:
        rag_avg = results["benchmarks"]["rag"]["avg_retrieval_time"]
        print(f"RAG Retrieval: {rag_avg:.4f}s avg retrieval")
    
    if results["benchmarks"]["database"]["select"]:
        db_avg = results["benchmarks"]["database"]["select"]["avg_query_time"]
        print(f"Database SELECT: {db_avg:.4f}s avg query")
    
    print(f"\nFull report saved to: {report_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
