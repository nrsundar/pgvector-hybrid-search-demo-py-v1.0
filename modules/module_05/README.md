# Module 05: Vector Indexing & Performance Optimization

## Overview
Optimizing vector search performance through advanced indexing strategies, query optimization, and scaling techniques for Aurora PostgreSQL pgvector implementations.

## Vector Index Types and Configuration

### 1. HNSW Index Optimization
```sql
-- HNSW (Hierarchical Navigable Small World) - Best for Aurora PostgreSQL
-- Optimal for high-dimensional vectors with good recall and performance

-- Documents embedding index (384 dimensions)
CREATE INDEX CONCURRENTLY idx_documents_embedding_hnsw ON documents 
USING hnsw (embedding vector_cosine_ops) 
WITH (
    m = 16,                    -- Number of bidirectional links for each node
    ef_construction = 64       -- Size of the candidate set during construction
);

-- Images visual embedding index (512 dimensions)  
CREATE INDEX CONCURRENTLY idx_images_visual_hnsw ON images 
USING hnsw (visual_embedding vector_cosine_ops) 
WITH (
    m = 16,
    ef_construction = 64
);

-- For higher accuracy, increase parameters (at cost of build time)
CREATE INDEX CONCURRENTLY idx_documents_embedding_hnsw_high_accuracy ON documents 
USING hnsw (embedding vector_cosine_ops) 
WITH (
    m = 32,                    -- Higher connectivity
    ef_construction = 128      -- Larger candidate set
);
```

### 2. IVFFlat Index for Large Datasets
```sql
-- IVFFlat (Inverted File Flat) - Better for very large datasets
-- Requires training phase but scales better than HNSW

-- Set number of lists (clusters) - typically sqrt(number_of_rows)
-- For 1M vectors, use ~1000 lists
-- For 10M vectors, use ~3162 lists

CREATE INDEX CONCURRENTLY idx_documents_embedding_ivfflat ON documents 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 1000);

CREATE INDEX CONCURRENTLY idx_images_visual_ivfflat ON images 
USING ivfflat (visual_embedding vector_cosine_ops) 
WITH (lists = 1000);

-- For exact results, you may need to adjust ivfflat.probes
SET ivfflat.probes = 10;  -- Default is 1, higher values = better recall

-- Check optimal probes value
SELECT 
    probes,
    recall,
    avg_query_time_ms
FROM (
    SELECT 
        p.probes,
        COUNT(CASE WHEN distance <= 0.1 THEN 1 END)::float / COUNT(*) as recall,
        AVG(query_time) as avg_query_time_ms
    FROM (
        VALUES (1), (5), (10), (20), (50), (100)
    ) p(probes)
    CROSS JOIN LATERAL (
        SELECT 
            embedding <=> %s as distance,
            extract(milliseconds from (clock_timestamp() - statement_timestamp())) as query_time
        FROM documents 
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT 100
    ) q
    GROUP BY p.probes
) results
ORDER BY probes;
```

### 3. Performance Monitoring and Analysis
```python
import time
import psycopg2
import numpy as np
from typing import List, Dict, Tuple

class VectorIndexPerformanceAnalyzer:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        
    def get_connection(self):
        return psycopg2.connect(**self.db_config)
    
    def benchmark_index_performance(
        self, 
        query_embedding: List[float],
        index_types: List[str] = ['hnsw', 'ivfflat', 'sequential'],
        num_queries: int = 100
    ) -> Dict[str, Dict]:
        """Benchmark different index types"""
        
        results = {}
        
        for index_type in index_types:
            print(f"Benchmarking {index_type} index...")
            
            query_times = []
            recall_scores = []
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                # Configure for index type
                if index_type == 'ivfflat':
                    cursor.execute("SET ivfflat.probes = 10;")
                elif index_type == 'hnsw':
                    cursor.execute("SET hnsw.ef_search = 100;")
                
                for _ in range(num_queries):
                    start_time = time.time()
                    
                    if index_type == 'sequential':
                        # Force sequential scan
                        cursor.execute("SET enable_indexscan = off;")
                        cursor.execute("SET enable_bitmapscan = off;")
                    
                    cursor.execute("""
                        SELECT id, 1 - (embedding <=> %s) AS similarity
                        FROM documents 
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %s
                        LIMIT 20;
                    """, (query_embedding, query_embedding))
                    
                    results_data = cursor.fetchall()
                    query_time = (time.time() - start_time) * 1000  # Convert to ms
                    query_times.append(query_time)
                    
                    # Reset settings
                    if index_type == 'sequential':
                        cursor.execute("SET enable_indexscan = on;")
                        cursor.execute("SET enable_bitmapscan = on;")
                
                results[index_type] = {
                    'avg_query_time_ms': np.mean(query_times),
                    'p95_query_time_ms': np.percentile(query_times, 95),
                    'p99_query_time_ms': np.percentile(query_times, 99),
                    'throughput_qps': 1000 / np.mean(query_times)
                }
                
            finally:
                cursor.close()
                conn.close()
        
        return results
    
    def analyze_index_usage(self) -> Dict:
        """Analyze index usage statistics"""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan as scans,
                    idx_tup_read as tuples_read,
                    idx_tup_fetch as tuples_fetched,
                    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                FROM pg_stat_user_indexes 
                WHERE indexname LIKE '%embedding%' 
                   OR indexname LIKE '%visual%'
                ORDER BY idx_scan DESC;
            """)
            
            index_stats = cursor.fetchall()
            
            return {
                'index_usage': [
                    {
                        'schema': row[0],
                        'table': row[1], 
                        'index': row[2],
                        'scans': row[3],
                        'tuples_read': row[4],
                        'tuples_fetched': row[5],
                        'size': row[6]
                    }
                    for row in index_stats
                ]
            }
            
        finally:
            cursor.close()
            conn.close()
```

### 4. Query Optimization Strategies
```sql
-- Optimize vector similarity queries with proper EXPLAIN analysis

-- Query 1: Basic similarity search with EXPLAIN
EXPLAIN (ANALYZE, BUFFERS, VERBOSE) 
SELECT 
    id, title, 
    1 - (embedding <=> '[0.1,0.2,0.3,...]'::vector) AS similarity
FROM documents 
WHERE embedding IS NOT NULL
ORDER BY embedding <=> '[0.1,0.2,0.3,...]'::vector
LIMIT 10;

-- Query 2: Filtered similarity search
EXPLAIN (ANALYZE, BUFFERS) 
SELECT 
    id, title, department,
    1 - (embedding <=> '[0.1,0.2,0.3,...]'::vector) AS similarity
FROM documents 
WHERE embedding IS NOT NULL
  AND department = 'Engineering'  -- Filter first, then vector search
  AND created_at > NOW() - INTERVAL '30 days'
ORDER BY embedding <=> '[0.1,0.2,0.3,...]'::vector
LIMIT 10;

-- Query 3: Combined text and vector search
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    id, title,
    1 - (embedding <=> '[0.1,0.2,0.3,...]'::vector) AS vector_similarity,
    ts_rank(to_tsvector('english', content), plainto_tsquery('machine learning')) AS text_rank
FROM documents 
WHERE embedding IS NOT NULL
  AND to_tsvector('english', content) @@ plainto_tsquery('machine learning')
ORDER BY 
    (1 - (embedding <=> '[0.1,0.2,0.3,...]'::vector)) * 0.7 +
    ts_rank(to_tsvector('english', content), plainto_tsquery('machine learning')) * 0.3 DESC
LIMIT 20;
```

### 5. Advanced Optimization Techniques
```python
def optimize_vector_queries():
    """Advanced query optimization techniques"""
    
    # 1. Embedding Normalization for Cosine Similarity
    def normalize_embedding(embedding: List[float]) -> List[float]:
        """Normalize embedding to unit vector for cosine similarity"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return (np.array(embedding) / norm).tolist()
    
    # 2. Quantization for Storage Efficiency
    def quantize_embedding(embedding: List[float], bits: int = 8) -> List[int]:
        """Quantize embedding to reduce storage size"""
        embedding_array = np.array(embedding)
        
        # Scale to quantization range
        min_val, max_val = embedding_array.min(), embedding_array.max()
        scale = (2**bits - 1) / (max_val - min_val)
        
        quantized = np.round((embedding_array - min_val) * scale).astype(int)
        return quantized.tolist(), min_val, max_val, scale
    
    # 3. Batch Vector Operations
    def batch_similarity_search(
        query_embeddings: List[List[float]], 
        table_name: str = 'documents',
        batch_size: int = 10
    ) -> List[List[Dict]]:
        """Perform batch similarity searches efficiently"""
        
        results = []
        conn = get_db_connection()
        
        try:
            cursor = conn.cursor()
            
            for i in range(0, len(query_embeddings), batch_size):
                batch = query_embeddings[i:i + batch_size]
                batch_results = []
                
                for embedding in batch:
                    cursor.execute(f"""
                        SELECT id, title, 1 - (embedding <=> %s) AS similarity
                        FROM {table_name}
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %s
                        LIMIT 10;
                    """, (embedding, embedding))
                    
                    batch_results.append([
                        {'id': row[0], 'title': row[1], 'similarity': row[2]}
                        for row in cursor.fetchall()
                    ])
                
                results.extend(batch_results)
        
        finally:
            cursor.close()
            conn.close()
        
        return results

# 4. Precomputed Similarity Matrices for Frequent Queries
def precompute_similarity_matrix(
    reference_embeddings: List[List[float]],
    table_name: str = 'documents'
):
    """Precompute similarities for frequently accessed embeddings"""
    
    conn = get_db_connection()
    
    try:
        cursor = conn.cursor()
        
        # Create precomputed similarities table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name}_similarities (
                ref_id INTEGER,
                target_id INTEGER,
                similarity FLOAT,
                computed_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (ref_id, target_id)
            );
        """)
        
        # Create index for fast lookups
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_similarities_ref 
            ON {table_name}_similarities (ref_id, similarity DESC);
        """)
        
        # Precompute similarities
        for ref_id, ref_embedding in enumerate(reference_embeddings):
            cursor.execute(f"""
                INSERT INTO {table_name}_similarities (ref_id, target_id, similarity)
                SELECT 
                    %s as ref_id,
                    id as target_id,
                    1 - (embedding <=> %s) as similarity
                FROM {table_name}
                WHERE embedding IS NOT NULL
                ON CONFLICT (ref_id, target_id) 
                DO UPDATE SET 
                    similarity = EXCLUDED.similarity,
                    computed_at = NOW();
            """, (ref_id, ref_embedding))
        
        conn.commit()
        print(f"Precomputed similarities for {len(reference_embeddings)} reference embeddings")
        
    finally:
        cursor.close()
        conn.close()
```

### 6. Scaling Strategies for Production
```sql
-- Connection pooling configuration for high-throughput vector search
-- Set in postgresql.conf or Aurora parameter group:

-- Connection settings
max_connections = 200
shared_buffers = '25% of total memory'
effective_cache_size = '75% of total memory'
work_mem = '256MB'  -- Important for vector operations
maintenance_work_mem = '2GB'

-- Vector-specific settings
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
parallel_setup_cost = 100
parallel_tuple_cost = 0.1

-- For HNSW index performance
random_page_cost = 1.1  -- SSD-optimized
seq_page_cost = 1.0

-- Monitor vector query performance
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Query to find slow vector operations
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    stddev_time,
    (total_time / sum(total_time) OVER ()) * 100 AS percentage
FROM pg_stat_statements 
WHERE query LIKE '%<=>%' OR query LIKE '%vector%'
ORDER BY total_time DESC
LIMIT 10;

-- Create materialized view for frequently accessed similar items
CREATE MATERIALIZED VIEW document_similarities AS
SELECT 
    d1.id as source_id,
    d1.title as source_title,
    d2.id as target_id,
    d2.title as target_title,
    1 - (d1.embedding <=> d2.embedding) as similarity
FROM documents d1
CROSS JOIN documents d2
WHERE d1.id != d2.id
  AND d1.embedding IS NOT NULL 
  AND d2.embedding IS NOT NULL
  AND 1 - (d1.embedding <=> d2.embedding) > 0.7  -- Only high similarity
ORDER BY d1.id, similarity DESC;

CREATE INDEX idx_document_similarities_source ON document_similarities (source_id, similarity DESC);

-- Refresh materialized view periodically
SELECT cron.schedule('refresh-similarities', '0 2 * * *', 'REFRESH MATERIALIZED VIEW document_similarities;');
```

### 7. Performance Testing Framework
```python
def comprehensive_performance_test():
    """Run comprehensive performance tests for vector operations"""
    
    analyzer = VectorIndexPerformanceAnalyzer({
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT', 5432),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    })
    
    # Generate test query embedding
    test_embedding = np.random.rand(384).tolist()
    
    # Benchmark different index types
    results = analyzer.benchmark_index_performance(
        query_embedding=test_embedding,
        index_types=['hnsw', 'ivfflat'],
        num_queries=50
    )
    
    print("\n=== Vector Index Performance Results ===")
    for index_type, metrics in results.items():
        print(f"\n{index_type.upper()} Index:")
        print(f"  Average Query Time: {metrics['avg_query_time_ms']:.2f}ms")
        print(f"  P95 Query Time: {metrics['p95_query_time_ms']:.2f}ms")
        print(f"  P99 Query Time: {metrics['p99_query_time_ms']:.2f}ms")
        print(f"  Throughput: {metrics['throughput_qps']:.1f} QPS")
    
    # Analyze index usage
    usage_stats = analyzer.analyze_index_usage()
    print("\n=== Index Usage Statistics ===")
    for idx in usage_stats['index_usage']:
        print(f"Index: {idx['index']}")
        print(f"  Scans: {idx['scans']}")
        print(f"  Size: {idx['size']}")
        print(f"  Efficiency: {idx['tuples_fetched'] / max(idx['tuples_read'], 1):.2f}")
        print()

if __name__ == "__main__":
    comprehensive_performance_test()
```

## Key Optimization Guidelines

1. **Choose the right index**: HNSW for most use cases, IVFFlat for very large datasets
2. **Monitor query performance**: Use EXPLAIN ANALYZE and pg_stat_statements
3. **Optimize embedding dimensions**: Balance between accuracy and performance
4. **Use connection pooling**: Essential for high-throughput vector applications
5. **Consider precomputation**: For frequently accessed similarity calculations
6. **Monitor resource usage**: Vector operations are memory-intensive

## Next Steps
Continue to Module 06 to implement advanced retrieval patterns and similarity thresholds.
