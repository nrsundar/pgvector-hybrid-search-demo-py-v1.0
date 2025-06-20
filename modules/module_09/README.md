# Module 09: Production Scaling & Optimization

## Overview
Implementing production-grade scaling strategies, performance optimization, and comprehensive monitoring for large-scale pgvector deployments on Aurora PostgreSQL.

## Production Scaling Architecture

### 1. Connection Pooling and Load Balancing
```python
import psycopg2
from psycopg2 import pool
import asyncio
import asyncpg
from typing import List, Dict, Optional
import time
import logging
from contextlib import contextmanager

class VectorDatabasePool:
    def __init__(self, config: Dict):
        self.config = config
        self.connection_pool = None
        self.async_pool = None
        
    def initialize_sync_pool(self):
        """Initialize synchronous connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config.get('min_connections', 5),
                maxconn=self.config.get('max_connections', 20),
                host=self.config['host'],
                port=self.config.get('port', 5432),
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                # Vector-specific optimizations
                options='-c work_mem=256MB -c effective_cache_size=4GB'
            )
            
            logging.info(f"Initialized connection pool with {self.config.get('max_connections', 20)} max connections")
            
        except Exception as e:
            logging.error(f"Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic cleanup"""
        connection = None
        
        try:
            connection = self.connection_pool.getconn()
            yield connection
            
        except Exception as e:
            if connection:
                connection.rollback()
            raise e
            
        finally:
            if connection:
                self.connection_pool.putconn(connection)
```

### 2. Advanced Performance Optimization
```sql
-- Create partitioned tables for large-scale vector storage
CREATE TABLE documents_partitioned (
    id BIGSERIAL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    department TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    partition_key INTEGER GENERATED ALWAYS AS (id % 10) STORED
) PARTITION BY HASH (partition_key);

-- Create partitions with vector indexes
DO $$
BEGIN
    FOR i IN 0..9 LOOP
        EXECUTE format('
            CREATE TABLE documents_partition_%s PARTITION OF documents_partitioned
            FOR VALUES WITH (modulus 10, remainder %s);
        ', i, i);
        
        -- Create vector index on each partition
        EXECUTE format('
            CREATE INDEX CONCURRENTLY idx_documents_partition_%s_embedding 
            ON documents_partition_%s 
            USING hnsw (embedding vector_cosine_ops) 
            WITH (m = 16, ef_construction = 64);
        ', i, i);
    END LOOP;
END
$$;

-- Function for distributed vector search across partitions
CREATE OR REPLACE FUNCTION distributed_vector_search(
    query_embedding vector(384),
    similarity_threshold float DEFAULT 0.7,
    result_limit int DEFAULT 100
)
RETURNS TABLE(
    doc_id bigint,
    title text,
    similarity_score float,
    partition_id int
) AS $$
DECLARE
    partition_query text;
    final_query text;
BEGIN
    -- Build dynamic query to search all partitions
    SELECT string_agg(
        format('
            SELECT id, title, 
                   1 - (embedding <=> %L) as similarity_score,
                   %s as partition_id
            FROM documents_partition_%s 
            WHERE embedding IS NOT NULL
            AND 1 - (embedding <=> %L) >= %s
        ', query_embedding, i, i, query_embedding, similarity_threshold),
        ' UNION ALL '
    ) INTO partition_query
    FROM generate_series(0, 9) i;
    
    -- Create final query with ordering and limit
    final_query := format('
        SELECT * FROM (%s) combined_results
        ORDER BY similarity_score DESC
        LIMIT %s
    ', partition_query, result_limit);
    
    -- Execute and return results
    RETURN QUERY EXECUTE final_query;
END;
$$ LANGUAGE plpgsql;
```

## Next Steps
Continue to Module 10 to implement real-world use case examples and complete applications.
