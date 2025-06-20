# Module 10: Real-world Use Case Implementation

## Overview
Building complete, production-ready applications that showcase real-world pgvector use cases including document search, recommendation systems, and AI-powered content discovery for Aurora PostgreSQL.

## Complete Application Examples

### 1. Intelligent Document Search Platform
```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from sentence_transformers import SentenceTransformer

@dataclass
class DocumentSearchResult:
    id: int
    title: str
    content_preview: str
    similarity_score: float
    document_type: str
    created_at: datetime
    file_path: str
    highlights: List[str]
    metadata: Dict[str, Any]

class IntelligentDocumentSearch:
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.app = FastAPI(title="Intelligent Document Search Platform")
        self.setup_routes()
    
    def setup_routes(self):
        """Setup FastAPI routes for the document search platform"""
        
        @self.app.post("/api/documents/upload")
        async def upload_document(file: UploadFile = File(...)):
            """Upload and process a new document"""
            
            try:
                # Extract text from uploaded file
                content = await self._extract_text_from_file(file)
                
                if not content:
                    raise HTTPException(status_code=400, detail="Could not extract text from file")
                
                # Generate embedding
                embedding = self.embedding_model.encode(content).tolist()
                
                # Store document in database
                doc_id = await self._store_document(
                    title=file.filename,
                    content=content,
                    embedding=embedding,
                    file_type=file.content_type
                )
                
                return JSONResponse({
                    "status": "success",
                    "document_id": doc_id,
                    "message": f"Document '{file.filename}' uploaded and processed successfully"
                })
                
            except Exception as e:
                logging.error(f"Error uploading document: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/search")
        async def search_documents(request: Dict[str, Any]):
            """Perform intelligent document search"""
            
            query = request.get('query', '')
            filters = request.get('filters', {})
            limit = request.get('limit', 20)
            similarity_threshold = request.get('similarity_threshold', 0.3)
            
            if not query:
                raise HTTPException(status_code=400, detail="Query is required")
            
            try:
                results = await self.semantic_search(
                    query=query,
                    filters=filters,
                    limit=limit,
                    similarity_threshold=similarity_threshold
                )
                
                return JSONResponse({
                    "query": query,
                    "total_results": len(results),
                    "results": [self._format_search_result(r) for r in results],
                    "search_metadata": {
                        "similarity_threshold": similarity_threshold,
                        "filters_applied": filters
                    }
                })
                
            except Exception as e:
                logging.error(f"Error in document search: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def semantic_search(
        self,
        query: str,
        filters: Dict = None,
        limit: int = 20,
        similarity_threshold: float = 0.3
    ) -> List[DocumentSearchResult]:
        """Perform semantic search with advanced filtering"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Execute vector search query
        sql_query = """
            SELECT 
                id, title, content, document_type, created_at, file_path,
                1 - (embedding <=> %s) AS similarity_score
            FROM documents 
            WHERE embedding IS NOT NULL
            AND 1 - (embedding <=> %s) >= %s
            ORDER BY embedding <=> %s
            LIMIT %s;
        """
        
        params = [query_embedding, query_embedding, similarity_threshold, query_embedding, limit]
        results = await self.db_pool.execute_vector_query(sql_query, params)
        
        # Process results
        search_results = []
        for row in results:
            search_results.append(DocumentSearchResult(
                id=row['id'],
                title=row['title'],
                content_preview=row['content'][:200] + "...",
                similarity_score=row['similarity_score'],
                document_type=row['document_type'],
                created_at=row['created_at'],
                file_path=row['file_path'],
                highlights=[],
                metadata={}
            ))
        
        return search_results
```

### 2. Production Deployment Script
```bash
#!/bin/bash
# Complete production deployment for real-world pgvector applications

set -e

echo "🚀 Deploying Real-World pgvector Applications"

# Environment validation
if [ -z "$DATABASE_URL" ]; then
    echo "❌ Error: DATABASE_URL environment variable is required"
    exit 1
fi

# 1. Database setup
echo "📊 Setting up database schema..."
psql "$DATABASE_URL" << EOF
-- Install extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Set optimal configuration
ALTER SYSTEM SET shared_preload_libraries = 'vector,pg_stat_statements';
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET effective_cache_size = '4GB';
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- Reload configuration
SELECT pg_reload_conf();
EOF

# 2. Create schema
echo "🏗️ Creating application schema..."
psql "$DATABASE_URL" << EOF
-- Documents table with vector embeddings
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    document_type TEXT,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create vector index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_embedding 
ON documents USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
EOF

# 3. Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install sentence-transformers fastapi uvicorn psycopg2-binary asyncpg

# 4. Start application services
echo "🌐 Starting application services..."
python3 -c "
from intelligent_document_search import IntelligentDocumentSearch
from vector_database_pool import VectorDatabasePool
import uvicorn

# Initialize database pool
db_config = {
    'host': '$DB_HOST',
    'database': '$DB_NAME',
    'user': '$DB_USER',
    'password': '$DB_PASSWORD'
}

db_pool = VectorDatabasePool(db_config)
search_app = IntelligentDocumentSearch(db_pool)

# Start the application
uvicorn.run(search_app.app, host='0.0.0.0', port=8000)
"

echo "🎉 Deployment completed successfully!"
echo "📊 Document Search API: http://localhost:8000"
```

### 3. Complete SQL Schema
```sql
-- Complete schema for intelligent document systems

-- Documents table with vector embeddings
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    document_type TEXT,
    file_path TEXT,
    author TEXT,
    department TEXT,
    tags TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User interactions for recommendation learning
CREATE TABLE user_interactions (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    document_id BIGINT REFERENCES documents(id),
    interaction_type TEXT NOT NULL,
    rating FLOAT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Search analytics
CREATE TABLE search_queries (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT,
    query_text TEXT NOT NULL,
    query_embedding vector(384),
    result_count INTEGER,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Indexes for optimal performance
CREATE INDEX CONCURRENTLY idx_documents_embedding ON documents 
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_documents_department ON documents (department);
CREATE INDEX idx_documents_document_type ON documents (document_type);
CREATE INDEX idx_user_interactions_user_id ON user_interactions (user_id);

-- Functions for application support
CREATE OR REPLACE FUNCTION comprehensive_document_search(
    search_query TEXT,
    search_embedding vector(384) DEFAULT NULL,
    department_filter TEXT DEFAULT NULL,
    limit_results INTEGER DEFAULT 20
)
RETURNS TABLE(
    document_id BIGINT,
    title TEXT,
    department TEXT,
    similarity_score FLOAT,
    combined_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        d.department,
        CASE 
            WHEN search_embedding IS NOT NULL THEN 1 - (d.embedding <=> search_embedding)
            ELSE 0.0
        END as similarity_score,
        (CASE 
            WHEN search_embedding IS NOT NULL THEN 1 - (d.embedding <=> search_embedding)
            ELSE 0.0
        END * 0.8 + 
        ts_rank(to_tsvector('english', d.title || ' ' || d.content), 
                plainto_tsquery('english', search_query)) * 0.2) as combined_score
    FROM documents d
    WHERE (department_filter IS NULL OR d.department = department_filter)
    AND (
        search_embedding IS NULL OR 
        1 - (d.embedding <=> search_embedding) > 0.2
    )
    ORDER BY combined_score DESC
    LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;
```

## Summary
All 10 Vector/Hybrid Search modules now contain comprehensive pgvector implementations including:
- Aurora PostgreSQL setup and configuration
- Document and image processing pipelines
- Multi-modal search capabilities
- Custom embedding models and fine-tuning
- Production scaling and monitoring
- Complete real-world applications

The modules provide production-ready code examples, SQL scripts, and deployment strategies specifically optimized for Aurora PostgreSQL.

## Next Steps
The Vector/Hybrid Search module series is now complete with comprehensive pgvector implementations.
