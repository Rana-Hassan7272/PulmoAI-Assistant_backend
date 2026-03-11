# Performance Benchmark Scripts

This directory contains scripts for benchmarking system performance.

## Available Scripts

### 1. `benchmark_ml_models.py`
Benchmarks ML model inference times:
- X-ray pneumonia detection
- Spirometry analysis
- Blood count disease prediction

**Usage:**
```bash
cd backend
python scripts/benchmark_ml_models.py [num_iterations]
```

**Example:**
```bash
python scripts/benchmark_ml_models.py 20
```

### 2. `benchmark_rag.py`
Benchmarks RAG retrieval performance.

**Usage:**
```bash
python scripts/benchmark_rag.py [num_iterations]
```

**Example:**
```bash
python scripts/benchmark_rag.py 20
```

### 3. `benchmark_database.py`
Benchmarks database query performance:
- SELECT queries
- INSERT queries

**Usage:**
```bash
python scripts/benchmark_database.py [num_iterations]
```

**Example:**
```bash
python scripts/benchmark_database.py 100
```

### 4. `run_all_benchmarks.py`
Runs all benchmarks and generates a comprehensive report.

**Usage:**
```bash
python scripts/run_all_benchmarks.py [num_iterations]
```

**Example:**
```bash
python scripts/run_all_benchmarks.py 10
```

## Output

- Benchmark results are printed to console
- Detailed reports are saved to `backend/performance_reports/` directory
- Reports are in JSON format with timestamps

## Performance Metrics API

You can also view real-time performance metrics via the API:

```bash
curl http://localhost:8000/metrics/performance
```

This returns aggregated statistics for:
- API response times
- ML model inference times
- RAG retrieval performance
- Database query performance

## Notes

- Benchmarks include warm-up runs to ensure accurate measurements
- Results include average, median, min, max, and P95 percentiles
- Database benchmarks use rollback to avoid filling the database
- ML model benchmarks require model files to be present
