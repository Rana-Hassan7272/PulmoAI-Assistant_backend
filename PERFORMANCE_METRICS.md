# Performance Metrics & Benchmarks

This document describes the performance monitoring system implemented in the Doctor Assistant project.

## Overview

The performance monitoring system tracks and logs metrics for:
- **API Response Times**: All HTTP requests
- **ML Model Inference**: X-ray, Spirometry, Blood Count models
- **RAG Retrieval**: Knowledge base query performance
- **Database Queries**: SELECT, INSERT, UPDATE, DELETE operations

## Features

### 1. Real-Time Performance Monitoring

All performance metrics are logged in real-time and stored in memory for quick access.

### 2. Performance API Endpoint

View current performance statistics:
```bash
GET /metrics/performance
```

Returns aggregated statistics including:
- Average, median, min, max response times
- P95 and P99 percentiles
- Slow request counts
- Per-model inference statistics

### 3. Automatic Logging

- **API Requests**: Automatically logged via middleware
- **ML Models**: Performance logged on each prediction
- **RAG System**: Retrieval times tracked automatically
- **Database**: Query performance tracked (can be added to specific queries)

### 4. Performance Thresholds

The system defines thresholds for "fast" and "slow" operations:
- API: Fast < 0.5s, Slow > 2.0s
- ML Models: Fast < 0.1s, Slow > 1.0s
- RAG: Fast < 0.05s, Slow > 0.5s
- Database: Fast < 0.01s, Slow > 0.1s

## Benchmark Scripts

### Running Benchmarks

All benchmark scripts are located in `backend/scripts/`:

#### 1. ML Model Benchmarks
```bash
cd backend
python scripts/benchmark_ml_models.py [iterations]
```

Benchmarks:
- X-ray pneumonia detection model
- Spirometry analysis model
- Blood count disease prediction model

Output: Average, median, min, max, P95 inference times

#### 2. RAG Benchmarks
```bash
python scripts/benchmark_rag.py [iterations]
```

Benchmarks:
- Retrieval time for various queries
- Documents retrieved per query
- Overall retrieval performance

#### 3. Database Benchmarks
```bash
python scripts/benchmark_database.py [iterations]
```

Benchmarks:
- SELECT query performance
- INSERT query performance
- Query time statistics

#### 4. Run All Benchmarks
```bash
python scripts/run_all_benchmarks.py [iterations]
```

Runs all benchmarks and generates a comprehensive JSON report.

## Performance Reports

Benchmark reports are saved to `backend/performance_reports/` directory in JSON format.

Each report includes:
- Timestamp
- Number of iterations
- Detailed statistics for each component
- Raw timing data

## Usage Examples

### View Current Performance Metrics

```bash
# Via API
curl http://localhost:8000/metrics/performance

# Or in Python
from app.core.performance import get_performance_stats
stats = get_performance_stats()
print(stats)
```

### Save Performance Report

```python
from app.core.performance import save_performance_report
report_path = save_performance_report()
print(f"Report saved to: {report_path}")
```

### Run Benchmarks

```bash
# Quick benchmark (10 iterations each)
python scripts/run_all_benchmarks.py

# Detailed benchmark (50 iterations each)
python scripts/run_all_benchmarks.py 50
```

## Performance Headers

API responses include performance headers:
- `X-Response-Time`: Request duration in seconds

## Logging

Performance metrics are logged at appropriate levels:
- **INFO**: Normal operations within thresholds
- **WARNING**: Slow operations exceeding thresholds
- **DEBUG**: Detailed timing information

## Integration

The performance monitoring system is automatically integrated:
- ✅ API middleware logs all requests
- ✅ ML models log inference times
- ✅ RAG system logs retrieval times
- ✅ Database queries can be tracked (manual integration)

## Future Enhancements

Potential improvements:
- Export metrics to Prometheus/Grafana
- Real-time dashboard
- Alerting for performance degradation
- Historical trend analysis
- Performance regression detection
