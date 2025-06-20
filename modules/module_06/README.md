# Module 06: Advanced Retrieval Patterns

## Overview
Implementing advanced retrieval patterns, similarity thresholds, and semantic filtering for production-grade pgvector applications in Aurora PostgreSQL.

## Advanced Retrieval Patterns

### 1. Multi-Stage Retrieval Pipeline
```python
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

class RetrievalStrategy(Enum):
    EXACT_MATCH = "exact"
    APPROXIMATE = "approximate"
    HYBRID_RERANK = "hybrid_rerank"
    SEMANTIC_FILTER = "semantic_filter"

@dataclass
class RetrievalResult:
    id: int
    score: float
    content: str
    metadata: Dict
    retrieval_stage: str

class AdvancedRetriever:
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.rerank_top_k = 100
        self.final_top_k = 20
    
    def multi_stage_retrieval(
        self,
        query_embedding: List[float],
        filters: Dict = None,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK
    ) -> List[RetrievalResult]:
        """Implement multi-stage retrieval for optimal performance and accuracy"""
        
        if strategy == RetrievalStrategy.HYBRID_RERANK:
            return self._hybrid_rerank_retrieval(query_embedding, filters)
        elif strategy == RetrievalStrategy.SEMANTIC_FILTER:
            return self._semantic_filter_retrieval(query_embedding, filters)
        else:
            return self._standard_retrieval(query_embedding, filters)
    
    def _hybrid_rerank_retrieval(
        self, 
        query_embedding: List[float], 
        filters: Dict = None
    ) -> List[RetrievalResult]:
        """Two-stage retrieval: fast approximate search + accurate reranking"""
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Stage 1: Fast approximate retrieval (larger candidate set)
            filter_clause = self._build_filter_clause(filters)
            
            cursor.execute(f"""
                SELECT 
                    id, title, content, embedding,
                    1 - (embedding <=> %s) as similarity
                FROM documents 
                WHERE embedding IS NOT NULL {filter_clause}
                ORDER BY embedding <=> %s
                LIMIT %s;
            """, (query_embedding, query_embedding, self.rerank_top_k))
            
            candidates = cursor.fetchall()
            
            # Stage 2: Accurate reranking with additional features
            reranked_results = self._rerank_candidates(
                candidates, query_embedding, filters
            )
            
            return reranked_results[:self.final_top_k]
            
        finally:
            cursor.close()
            conn.close()
    
    def _rerank_candidates(
        self, 
        candidates: List, 
        query_embedding: List[float],
        filters: Dict = None
    ) -> List[RetrievalResult]:
        """Advanced reranking with multiple signals"""
        
        reranked = []
        query_vector = np.array(query_embedding)
        
        for candidate in candidates:
            id_, title, content, embedding, initial_similarity = candidate
            
            # Calculate multiple similarity metrics
            embedding_vector = np.array(embedding)
            
            # Cosine similarity (more precise calculation)
            cosine_sim = np.dot(query_vector, embedding_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(embedding_vector)
            )
            
            # Euclidean distance (normalized)
            euclidean_dist = np.linalg.norm(query_vector - embedding_vector)
            euclidean_sim = 1 / (1 + euclidean_dist)
            
            # Content length normalization
            content_length_factor = min(1.0, len(content) / 1000)
            
            # Combined score with weights
            final_score = (
                cosine_sim * 0.7 +
                euclidean_sim * 0.2 +
                content_length_factor * 0.1
            )
            
            # Apply similarity threshold
            if final_score >= self.similarity_threshold:
                reranked.append(RetrievalResult(
                    id=id_,
                    score=final_score,
                    content=content[:200] + "..." if len(content) > 200 else content,
                    metadata={
                        'title': title,
                        'cosine_similarity': cosine_sim,
                        'euclidean_similarity': euclidean_sim,
                        'content_length': len(content)
                    },
                    retrieval_stage='rerank'
                ))
        
        # Sort by final score
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked
```

### 2. Semantic Filtering and Clustering
```python
def semantic_clustering_retrieval(
    query_embedding: List[float],
    cluster_threshold: float = 0.8,
    max_clusters: int = 5
) -> Dict[str, List[RetrievalResult]]:
    """Group similar results into semantic clusters"""
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Get initial candidates
        cursor.execute("""
            SELECT 
                id, title, content, embedding,
                1 - (embedding <=> %s) as similarity
            FROM documents 
            WHERE embedding IS NOT NULL
            AND 1 - (embedding <=> %s) > %s
            ORDER BY embedding <=> %s
            LIMIT 100;
        """, (query_embedding, query_embedding, cluster_threshold, query_embedding))
        
        candidates = cursor.fetchall()
        
        # Perform clustering
        clusters = perform_semantic_clustering(candidates, cluster_threshold)
        
        return clusters
        
    finally:
        cursor.close()
        conn.close()

def perform_semantic_clustering(
    candidates: List, 
    threshold: float
) -> Dict[str, List[RetrievalResult]]:
    """Cluster semantically similar results"""
    
    embeddings = np.array([candidate[3] for candidate in candidates])
    clusters = {}
    cluster_id = 0
    assigned = set()
    
    for i, candidate in enumerate(candidates):
        if i in assigned:
            continue
            
        # Start new cluster
        cluster_key = f"cluster_{cluster_id}"
        clusters[cluster_key] = []
        
        # Add current candidate
        clusters[cluster_key].append(RetrievalResult(
            id=candidate[0],
            score=candidate[4],
            content=candidate[2][:200] + "...",
            metadata={'title': candidate[1]},
            retrieval_stage='cluster'
        ))
        assigned.add(i)
        
        # Find similar candidates for this cluster
        current_embedding = embeddings[i]
        
        for j, other_candidate in enumerate(candidates):
            if j in assigned:
                continue
                
            # Calculate similarity between candidates
            other_embedding = embeddings[j]
            similarity = np.dot(current_embedding, other_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(other_embedding)
            )
            
            if similarity > threshold:
                clusters[cluster_key].append(RetrievalResult(
                    id=other_candidate[0],
                    score=other_candidate[4],
                    content=other_candidate[2][:200] + "...",
                    metadata={'title': other_candidate[1]},
                    retrieval_stage='cluster'
                ))
                assigned.add(j)
        
        cluster_id += 1
    
    return clusters
```

### 3. Adaptive Similarity Thresholds
```sql
-- Dynamic similarity threshold based on query complexity
CREATE OR REPLACE FUNCTION adaptive_similarity_search(
    query_embedding vector(384),
    base_threshold float DEFAULT 0.7,
    max_results int DEFAULT 20,
    min_results int DEFAULT 5
)
RETURNS TABLE(
    doc_id int,
    title text,
    similarity_score float,
    confidence_level text
) AS $$
DECLARE
    result_count int;
    adjusted_threshold float;
BEGIN
    -- Start with base threshold
    adjusted_threshold := base_threshold;
    
    -- Loop to find optimal threshold
    LOOP
        -- Count results at current threshold
        SELECT COUNT(*) INTO result_count
        FROM documents 
        WHERE embedding IS NOT NULL
        AND 1 - (embedding <=> query_embedding) >= adjusted_threshold;
        
        -- Exit conditions
        IF result_count >= min_results AND result_count <= max_results THEN
            EXIT; -- Perfect range
        ELSIF result_count > max_results THEN
            adjusted_threshold := adjusted_threshold + 0.05; -- Increase threshold
        ELSIF result_count < min_results THEN
            adjusted_threshold := adjusted_threshold - 0.05; -- Decrease threshold
        END IF;
        
        -- Safety exit
        IF adjusted_threshold <= 0.3 OR adjusted_threshold >= 0.95 THEN
            EXIT;
        END IF;
    END LOOP;
    
    -- Return results with confidence levels
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        (1 - (d.embedding <=> query_embedding))::float as similarity_score,
        CASE 
            WHEN 1 - (d.embedding <=> query_embedding) >= 0.9 THEN 'high'
            WHEN 1 - (d.embedding <=> query_embedding) >= 0.7 THEN 'medium'
            ELSE 'low'
        END as confidence_level
    FROM documents d
    WHERE d.embedding IS NOT NULL
    AND 1 - (d.embedding <=> query_embedding) >= adjusted_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT max_results;
    
END;
$$ LANGUAGE plpgsql;

-- Test adaptive threshold function
SELECT * FROM adaptive_similarity_search(
    (SELECT embedding FROM documents WHERE id = 1)::vector(384),
    0.7,
    15,
    3
);
```

### 4. Multi-Modal Retrieval Patterns
```python
class MultiModalRetriever:
    def __init__(self):
        self.text_weight = 0.6
        self.visual_weight = 0.4
        self.cross_modal_boost = 1.2
    
    def cross_modal_retrieval(
        self,
        text_query: Optional[str] = None,
        image_query_embedding: Optional[List[float]] = None,
        mode: str = "complementary"
    ) -> List[RetrievalResult]:
        """Implement cross-modal retrieval patterns"""
        
        if mode == "complementary":
            return self._complementary_retrieval(text_query, image_query_embedding)
        elif mode == "reinforcing":
            return self._reinforcing_retrieval(text_query, image_query_embedding)
        else:
            return self._independent_retrieval(text_query, image_query_embedding)
    
    def _complementary_retrieval(
        self, 
        text_query: str, 
        image_embedding: List[float]
    ) -> List[RetrievalResult]:
        """Find content that complements the query across modalities"""
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Find documents with text similarity
            text_embedding = generate_text_embedding(text_query)
            
            cursor.execute("""
                WITH text_matches AS (
                    SELECT id, 'document' as type, title, content,
                           1 - (embedding <=> %s) as text_sim,
                           0 as visual_sim
                    FROM documents 
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT 50
                ),
                image_matches AS (
                    SELECT id, 'image' as type, filename as title, description as content,
                           0 as text_sim,
                           1 - (visual_embedding <=> %s) as visual_sim
                    FROM images 
                    WHERE visual_embedding IS NOT NULL
                    ORDER BY visual_embedding <=> %s
                    LIMIT 50
                )
                SELECT * FROM text_matches
                UNION ALL
                SELECT * FROM image_matches
                ORDER BY (text_sim * %s + visual_sim * %s) DESC
                LIMIT 20;
            """, (
                text_embedding, text_embedding,
                image_embedding, image_embedding,
                self.text_weight, self.visual_weight
            ))
            
            results = cursor.fetchall()
            
            return [
                RetrievalResult(
                    id=row[0],
                    score=row[4] * self.text_weight + row[5] * self.visual_weight,
                    content=row[3][:200] + "..." if row[3] else "",
                    metadata={
                        'type': row[1],
                        'title': row[2],
                        'text_similarity': row[4],
                        'visual_similarity': row[5]
                    },
                    retrieval_stage='cross_modal'
                )
                for row in results
            ]
            
        finally:
            cursor.close()
            conn.close()
```

### 5. Performance-Optimized Retrieval
```sql
-- Materialized view for frequently accessed similarity patterns
CREATE MATERIALIZED VIEW IF NOT EXISTS popular_document_similarities AS
SELECT 
    d1.id as source_id,
    d2.id as target_id,
    1 - (d1.embedding <=> d2.embedding) as similarity,
    d1.title as source_title,
    d2.title as target_title
FROM documents d1
CROSS JOIN documents d2
WHERE d1.id != d2.id
  AND d1.embedding IS NOT NULL 
  AND d2.embedding IS NOT NULL
  AND 1 - (d1.embedding <=> d2.embedding) > 0.8  -- High similarity only
ORDER BY similarity DESC;

-- Index for fast similarity lookups
CREATE INDEX idx_popular_similarities_source ON popular_document_similarities (source_id, similarity DESC);
CREATE INDEX idx_popular_similarities_target ON popular_document_similarities (target_id, similarity DESC);

-- Function for cached similarity retrieval
CREATE OR REPLACE FUNCTION get_similar_documents_cached(
    source_doc_id int,
    min_similarity float DEFAULT 0.7,
    result_limit int DEFAULT 10
)
RETURNS TABLE(
    document_id int,
    title text,
    similarity_score float,
    cache_hit boolean
) AS $$
BEGIN
    -- Try cached results first
    RETURN QUERY
    SELECT 
        ps.target_id,
        ps.target_title,
        ps.similarity,
        true as cache_hit
    FROM popular_document_similarities ps
    WHERE ps.source_id = source_doc_id
    AND ps.similarity >= min_similarity
    ORDER BY ps.similarity DESC
    LIMIT result_limit;
    
    -- If no cached results, fall back to real-time calculation
    IF NOT FOUND THEN
        RETURN QUERY
        SELECT 
            d.id,
            d.title,
            (1 - (d.embedding <=> (SELECT embedding FROM documents WHERE id = source_doc_id)))::float,
            false as cache_hit
        FROM documents d
        WHERE d.id != source_doc_id
        AND d.embedding IS NOT NULL
        AND 1 - (d.embedding <=> (SELECT embedding FROM documents WHERE id = source_doc_id)) >= min_similarity
        ORDER BY d.embedding <=> (SELECT embedding FROM documents WHERE id = source_doc_id)
        LIMIT result_limit;
    END IF;
    
END;
$$ LANGUAGE plpgsql;

-- Refresh cache periodically
SELECT cron.schedule('refresh-similarity-cache', '0 3 * * *', 
    'REFRESH MATERIALIZED VIEW popular_document_similarities;');
```

### 6. Advanced Filtering and Faceting
```python
def faceted_vector_search(
    query_embedding: List[float],
    facets: Dict[str, List[str]] = None,
    date_range: Tuple[str, str] = None,
    content_types: List[str] = None
) -> Dict[str, any]:
    """Implement faceted search with vector similarity"""
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Build dynamic query with facets
        where_clauses = ["embedding IS NOT NULL"]
        params = [query_embedding, query_embedding]
        
        if facets:
            for facet_field, facet_values in facets.items():
                if facet_values:
                    placeholders = ','.join(['%s'] * len(facet_values))
                    where_clauses.append(f"{facet_field} IN ({placeholders})")
                    params.extend(facet_values)
        
        if date_range:
            where_clauses.append("created_at BETWEEN %s AND %s")
            params.extend(date_range)
        
        where_clause = " AND ".join(where_clauses)
        
        # Main search query
        cursor.execute(f"""
            SELECT 
                id, title, content, department, author, created_at,
                1 - (embedding <=> %s) as similarity
            FROM documents 
            WHERE {where_clause}
            ORDER BY embedding <=> %s
            LIMIT 50;
        """, params)
        
        results = cursor.fetchall()
        
        # Calculate facet counts
        facet_counts = calculate_facet_counts(cursor, where_clause, params[2:])
        
        return {
            'results': [
                {
                    'id': row[0],
                    'title': row[1],
                    'content': row[2][:200] + "...",
                    'department': row[3],
                    'author': row[4],
                    'created_at': row[5].isoformat(),
                    'similarity': row[6]
                }
                for row in results
            ],
            'facets': facet_counts,
            'total_count': len(results)
        }
        
    finally:
        cursor.close()
        conn.close()

def calculate_facet_counts(cursor, base_where_clause: str, base_params: List) -> Dict:
    """Calculate facet counts for search results"""
    
    facet_queries = {
        'departments': "SELECT department, COUNT(*) FROM documents WHERE {} GROUP BY department",
        'authors': "SELECT author, COUNT(*) FROM documents WHERE {} GROUP BY author",
        'date_ranges': """
            SELECT 
                CASE 
                    WHEN created_at > NOW() - INTERVAL '7 days' THEN 'last_week'
                    WHEN created_at > NOW() - INTERVAL '30 days' THEN 'last_month'
                    WHEN created_at > NOW() - INTERVAL '90 days' THEN 'last_quarter'
                    ELSE 'older'
                END as date_range,
                COUNT(*)
            FROM documents 
            WHERE {} 
            GROUP BY date_range
        """
    }
    
    facet_counts = {}
    
    for facet_name, query_template in facet_queries.items():
        cursor.execute(query_template.format(base_where_clause), base_params)
        facet_counts[facet_name] = [
            {'value': row[0], 'count': row[1]}
            for row in cursor.fetchall()
        ]
    
    return facet_counts
```

## Testing Advanced Retrieval
```python
def test_advanced_retrieval_patterns():
    """Test suite for advanced retrieval patterns"""
    
    # Test data
    test_embedding = np.random.rand(384).tolist()
    
    print("=== Testing Advanced Retrieval Patterns ===\n")
    
    # Test 1: Multi-stage retrieval
    retriever = AdvancedRetriever(similarity_threshold=0.7)
    results = retriever.multi_stage_retrieval(
        test_embedding,
        strategy=RetrievalStrategy.HYBRID_RERANK
    )
    print(f"Multi-stage retrieval: {len(results)} results")
    
    # Test 2: Semantic clustering
    clusters = semantic_clustering_retrieval(test_embedding)
    print(f"Semantic clustering: {len(clusters)} clusters")
    
    # Test 3: Adaptive thresholds (SQL function test)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM adaptive_similarity_search(%s, 0.7, 15, 3);", 
                      (test_embedding,))
        adaptive_results = cursor.fetchall()
        print(f"Adaptive threshold: {len(adaptive_results)} results")
    finally:
        cursor.close()
        conn.close()
    
    # Test 4: Faceted search
    faceted_results = faceted_vector_search(
        test_embedding,
        facets={'department': ['Engineering', 'Marketing']},
        content_types=['document']
    )
    print(f"Faceted search: {faceted_results['total_count']} results")
    print(f"Available facets: {list(faceted_results['facets'].keys())}")

if __name__ == "__main__":
    test_advanced_retrieval_patterns()
```

## Performance Monitoring
```sql
-- Monitor retrieval pattern performance
CREATE TABLE IF NOT EXISTS retrieval_metrics (
    id SERIAL PRIMARY KEY,
    query_type TEXT,
    execution_time_ms FLOAT,
    result_count INTEGER,
    similarity_threshold FLOAT,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Function to log retrieval performance
CREATE OR REPLACE FUNCTION log_retrieval_performance(
    query_type TEXT,
    start_time TIMESTAMP,
    result_count INTEGER,
    threshold FLOAT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO retrieval_metrics (query_type, execution_time_ms, result_count, similarity_threshold)
    VALUES (
        query_type,
        EXTRACT(milliseconds FROM (NOW() - start_time)),
        result_count,
        threshold
    );
END;
$$ LANGUAGE plpgsql;
```

## Next Steps
Continue to Module 08 to implement embedding models fine-tuning and custom similarity functions for domain-specific applications.
