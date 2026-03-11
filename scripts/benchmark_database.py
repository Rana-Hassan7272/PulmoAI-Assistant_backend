"""
Database Query Performance Benchmark Script

Benchmarks database query performance for common operations.
"""
import sys
import time
import statistics
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import SessionLocal, get_db
from app.db_models.patient import Patient
from app.db_models.user import User
from app.db_models.visit import Visit
from app.core.performance import log_db_query


def benchmark_select_queries(num_iterations: int = 100):
    """Benchmark SELECT query performance."""
    print("\n" + "="*60)
    print("BENCHMARKING DATABASE SELECT QUERIES")
    print("="*60)
    
    db = SessionLocal()
    times = []
    
    try:
        # Get count of records first
        patient_count = db.query(Patient).count()
        user_count = db.query(User).count()
        visit_count = db.query(Visit).count()
        
        print(f"Database status: {patient_count} patients, {user_count} users, {visit_count} visits")
        print(f"Running {num_iterations} SELECT query iterations...\n")
        
        # Benchmark: Get all patients
        print("Benchmarking: SELECT * FROM patients")
        patient_times = []
        for i in range(num_iterations):
            start = time.time()
            patients = db.query(Patient).all()
            duration = time.time() - start
            patient_times.append(duration)
            log_db_query("SELECT", duration, "patients")
        
        avg_patient = statistics.mean(patient_times)
        print(f"  Avg time: {avg_patient:.4f}s")
        
        # Benchmark: Get patient by ID (if patients exist)
        if patient_count > 0:
            print("Benchmarking: SELECT * FROM patients WHERE id = ?")
            patient_id_times = []
            test_id = db.query(Patient).first().id
            for i in range(num_iterations):
                start = time.time()
                patient = db.query(Patient).filter(Patient.id == test_id).first()
                duration = time.time() - start
                patient_id_times.append(duration)
                log_db_query("SELECT", duration, "patients")
            
            avg_patient_id = statistics.mean(patient_id_times)
            print(f"  Avg time: {avg_patient_id:.4f}s")
        
        # Benchmark: Get all visits
        print("Benchmarking: SELECT * FROM visits")
        visit_times = []
        for i in range(num_iterations):
            start = time.time()
            visits = db.query(Visit).all()
            duration = time.time() - start
            visit_times.append(duration)
            log_db_query("SELECT", duration, "visits")
        
        avg_visit = statistics.mean(visit_times)
        print(f"  Avg time: {avg_visit:.4f}s")
        
        # Overall statistics
        all_times = patient_times + visit_times
        if patient_count > 0:
            all_times.extend(patient_id_times)
        
        overall_stats = {
            "total_queries": len(all_times),
            "avg_query_time": statistics.mean(all_times),
            "median_query_time": statistics.median(all_times),
            "min_query_time": min(all_times),
            "max_query_time": max(all_times),
            "p95_query_time": sorted(all_times)[int(len(all_times) * 0.95)] if all_times else 0,
        }
        
        print("\n" + "-"*60)
        print("OVERALL SELECT QUERY RESULTS:")
        print(f"  Total queries: {overall_stats['total_queries']}")
        print(f"  Avg query time: {overall_stats['avg_query_time']:.4f}s")
        print(f"  Median query time: {overall_stats['median_query_time']:.4f}s")
        print(f"  P95 query time: {overall_stats['p95_query_time']:.4f}s")
        print("-"*60)
        
        return overall_stats
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()


def benchmark_insert_queries(num_iterations: int = 50):
    """Benchmark INSERT query performance."""
    print("\n" + "="*60)
    print("BENCHMARKING DATABASE INSERT QUERIES")
    print("="*60)
    
    db = SessionLocal()
    times = []
    
    try:
        print(f"Running {num_iterations} INSERT query iterations...\n")
        
        for i in range(num_iterations):
            start = time.time()
            
            # Create test patient
            patient = Patient(
                name=f"Test Patient {i}",
                age=30 + (i % 50),
                gender="Male" if i % 2 == 0 else "Female",
                smoker=False
            )
            db.add(patient)
            db.flush()  # Get ID without committing
            
            duration = time.time() - start
            times.append(duration)
            log_db_query("INSERT", duration, "patients")
            
            # Rollback to avoid filling database
            db.rollback()
        
        stats = {
            "total_inserts": len(times),
            "avg_insert_time": statistics.mean(times),
            "median_insert_time": statistics.median(times),
            "min_insert_time": min(times),
            "max_insert_time": max(times),
        }
        
        print(f"  Avg INSERT time: {stats['avg_insert_time']:.4f}s")
        print(f"  Median INSERT time: {stats['median_insert_time']:.4f}s")
        
        return stats
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()


def main():
    """Run database benchmarks."""
    print("\n" + "="*60)
    print("DATABASE QUERY PERFORMANCE BENCHMARK")
    print("="*60)
    print("\nThis script benchmarks database query performance.\n")
    
    num_iterations = 100
    if len(sys.argv) > 1:
        try:
            num_iterations = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of iterations: {sys.argv[1]}. Using default: 100")
    
    # Benchmark SELECT queries
    select_results = benchmark_select_queries(num_iterations)
    
    # Benchmark INSERT queries (fewer iterations)
    insert_results = benchmark_insert_queries(num_iterations // 2)
    
    print("\n" + "="*60)
    print("Benchmark completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
