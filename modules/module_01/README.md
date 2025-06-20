# Module 01: pgvector Extension Setup & Vector Fundamentals

## Overview
Setting up Amazon Aurora PostgreSQL 16 with pgvector extension for multi-modal vector search capabilities.

## Objectives
- Configure Aurora PostgreSQL with pgvector extension
- Understand vector data types and operations
- Set up database schema for multi-modal search
- Configure connection settings for vector operations

## Aurora PostgreSQL Setup

### 1. Enable pgvector Extension
```sql
-- Connect to your Aurora database
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check pgvector version
SELECT vector_version();
```

### 2. Database Schema for Multi-Modal Search
```sql
-- Create documents table with text embeddings
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    s3_key TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    -- Text embedding vector (384 dimensions for all-MiniLM-L6-v2)
    embedding vector(384),
    -- Metadata for filtering
    department TEXT,
    author TEXT,
    tags TEXT[]
);

-- Create images table with visual embeddings
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    description TEXT,
    s3_key TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    -- Visual embedding vector (512 dimensions for ResNet)
    visual_embedding vector(512),
    -- Image metadata
    width INTEGER,
    height INTEGER,
    file_size BIGINT,
    content_type TEXT
);

-- Create hybrid search results table for caching
CREATE TABLE search_cache (
    cache_key TEXT PRIMARY KEY,
    query_text TEXT,
    query_image_s3_key TEXT,
    results JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);
```

### 3. Vector Indexes for Performance
```sql
-- Create HNSW index for document embeddings (best for Aurora)
CREATE INDEX idx_documents_embedding ON documents 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Create HNSW index for image embeddings
CREATE INDEX idx_images_visual_embedding ON images 
USING hnsw (visual_embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Create supporting indexes for filtering
CREATE INDEX idx_documents_department ON documents (department);
CREATE INDEX idx_documents_created_at ON documents (created_at);
CREATE INDEX idx_images_content_type ON images (content_type);
```

### 4. Aurora Parameter Configuration
```sql
-- Optimize for vector operations
-- These should be set in your parameter group:
shared_preload_libraries = 'vector'
max_parallel_workers_per_gather = 2
work_mem = '256MB'
effective_cache_size = '75% of total memory'
shared_buffers = '25% of total memory'
```

## Vector Operations Fundamentals

### Basic Vector Operations
```sql
-- Calculate cosine distance between vectors
SELECT embedding <=> '[0.1,0.2,0.3]'::vector AS cosine_distance 
FROM documents WHERE id = 1;

-- Calculate cosine similarity (1 - cosine distance)
SELECT 1 - (embedding <=> '[0.1,0.2,0.3]'::vector) AS cosine_similarity 
FROM documents WHERE id = 1;

-- Find most similar documents
SELECT id, title, 1 - (embedding <=> '[0.1,0.2,0.3]'::vector) AS similarity
FROM documents 
WHERE embedding IS NOT NULL
ORDER BY embedding <=> '[0.1,0.2,0.3]'::vector
LIMIT 10;
```

## Performance Testing
```sql
-- Test vector search performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT id, title, embedding <=> '[0.1,0.2,0.3]'::vector AS distance
FROM documents 
ORDER BY embedding <=> '[0.1,0.2,0.3]'::vector
LIMIT 10;

-- Monitor index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM pg_stat_user_indexes 
WHERE indexname LIKE '%embedding%';
```

## Next Steps
Proceed to Module 02 to learn document processing and text embedding generation.
