"""
RAG System Performance Benchmark Script

Benchmarks retrieval performance for the RAG system.
"""
import sys
import time
import statistics
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.rag.rag_agent import get_rag_agent


def benchmark_rag_retrieval(num_iterations: int = 10):
    """Benchmark RAG retrieval performance."""
    print("\n" + "="*60)
    print("BENCHMARKING RAG RETRIEVAL SYSTEM")
    print("="*60)
    
    # Test queries
    test_queries = [
        "What are the symptoms of pneumonia?",
        "How to treat bacterial infection?",
        "What is the recommended dosage for antibiotics?",
        "What are the side effects of medication?",
        "How to diagnose respiratory disease?",
    ]
    
    try:
        rag_agent = get_rag_agent()
        
        # Check if RAG system is ready
        stats = rag_agent.get_stats()
        if stats.get('total_documents', 0) == 0:
            print("WARNING: RAG system has no documents indexed.")
            print("Please add documents to the knowledge base first.")
            return None
        
        print(f"RAG System Status: {stats.get('total_documents', 0)} document chunks indexed")
        print(f"Running {num_iterations} retrieval iterations per query...\n")
        
        all_times = []
        all_docs_retrieved = []
        
        for query_idx, query in enumerate(test_queries, 1):
            print(f"Query {query_idx}/{len(test_queries)}: '{query[:50]}...'")
            
            query_times = []
            query_docs = []
            
            for i in range(num_iterations):
                start = time.time()
                context = rag_agent.retrieve_context(query, k=5, min_similarity=0.3)
                duration = time.time() - start
                
                query_times.append(duration)
                docs_count = context.count("[") if context else 0
                query_docs.append(docs_count)
                all_times.append(duration)
                all_docs_retrieved.append(docs_count)
            
            avg_time = statistics.mean(query_times)
            avg_docs = statistics.mean(query_docs)
            print(f"  Avg retrieval time: {avg_time:.4f}s, Avg documents: {avg_docs:.1f}")
        
        # Overall statistics
        overall_stats = {
            "total_retrievals": len(all_times),
            "avg_retrieval_time": statistics.mean(all_times),
            "median_retrieval_time": statistics.median(all_times),
            "min_retrieval_time": min(all_times),
            "max_retrieval_time": max(all_times),
            "std_retrieval_time": statistics.stdev(all_times) if len(all_times) > 1 else 0,
            "p95_retrieval_time": sorted(all_times)[int(len(all_times) * 0.95)] if all_times else 0,
            "avg_documents_retrieved": statistics.mean(all_docs_retrieved) if all_docs_retrieved else 0,
        }
        
        print("\n" + "-"*60)
        print("OVERALL RESULTS:")
        print(f"  Total retrievals: {overall_stats['total_retrievals']}")
        print(f"  Avg retrieval time: {overall_stats['avg_retrieval_time']:.4f}s")
        print(f"  Median retrieval time: {overall_stats['median_retrieval_time']:.4f}s")
        print(f"  P95 retrieval time: {overall_stats['p95_retrieval_time']:.4f}s")
        print(f"  Avg documents retrieved: {overall_stats['avg_documents_retrieved']:.1f}")
        print("-"*60)
        
        return overall_stats
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run RAG benchmark."""
    print("\n" + "="*60)
    print("RAG SYSTEM PERFORMANCE BENCHMARK")
    print("="*60)
    print("\nThis script benchmarks retrieval performance for the RAG system.\n")
    
    num_iterations = 10
    if len(sys.argv) > 1:
        try:
            num_iterations = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of iterations: {sys.argv[1]}. Using default: 10")
    
    results = benchmark_rag_retrieval(num_iterations)
    
    if results:
        print("\n" + "="*60)
        print("Benchmark completed successfully!")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("Benchmark failed. Please check RAG system setup.")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
