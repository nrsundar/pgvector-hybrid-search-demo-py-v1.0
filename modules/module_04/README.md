# Module 04: Hybrid Search Implementation

## Overview
Implementing hybrid search that combines text and visual similarity scoring for comprehensive multi-modal search capabilities in Aurora PostgreSQL.

## Hybrid Search Architecture

### 1. Unified Search Interface
```python
import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum

class SearchMode(Enum):
    TEXT_ONLY = "text_only"
    VISUAL_ONLY = "visual_only"
    HYBRID = "hybrid"
    CROSS_MODAL = "cross_modal"

@dataclass
class SearchResult:
    id: int
    type: str  # 'document' or 'image'
    title: str
    content: Optional[str]
    s3_key: Optional[str]
    text_similarity: float
    visual_similarity: float
    hybrid_score: float
    metadata: Dict

class HybridSearchEngine:
    def __init__(self):
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.visual_model = self._load_visual_model()
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
    
    def _load_visual_model(self):
        """Load ResNet model for visual embeddings"""
        model = resnet50(pretrained=True)
        model.fc = torch.nn.Identity()
        model.eval()
        return model
```

### 2. Weighted Hybrid Scoring
```python
def calculate_hybrid_score(
    text_similarity: float,
    visual_similarity: float,
    text_weight: float = 0.6,
    visual_weight: float = 0.4,
    boost_factor: float = 1.0
) -> float:
    """Calculate weighted hybrid score combining text and visual similarities"""
    
    # Normalize weights
    total_weight = text_weight + visual_weight
    text_weight_norm = text_weight / total_weight
    visual_weight_norm = visual_weight / total_weight
    
    # Calculate weighted score
    hybrid_score = (
        text_similarity * text_weight_norm + 
        visual_similarity * visual_weight_norm
    ) * boost_factor
    
    return min(hybrid_score, 1.0)  # Cap at 1.0

def adaptive_weight_calculation(
    query_text: str,
    has_visual_query: bool,
    content_type_preference: str = "balanced"
) -> tuple:
    """Dynamically adjust weights based on query characteristics"""
    
    # Base weights
    text_weight = 0.5
    visual_weight = 0.5
    
    # Adjust based on query characteristics
    if len(query_text.split()) > 10:  # Long text query
        text_weight = 0.7
        visual_weight = 0.3
    elif len(query_text.split()) <= 3:  # Short query
        text_weight = 0.4
        visual_weight = 0.6
    
    # Adjust based on visual query presence
    if has_visual_query:
        visual_weight += 0.2
        text_weight -= 0.2
    
    # Preference-based adjustment
    if content_type_preference == "text_heavy":
        text_weight = 0.8
        visual_weight = 0.2
    elif content_type_preference == "visual_heavy":
        text_weight = 0.2
        visual_weight = 0.8
    
    return max(0.1, text_weight), max(0.1, visual_weight)
```

### 3. Multi-Modal Search Implementation
```python
def hybrid_search(
    query_text: str,
    query_image_s3_key: Optional[str] = None,
    search_mode: SearchMode = SearchMode.HYBRID,
    limit: int = 20,
    content_types: List[str] = ["documents", "images"],
    filters: Dict = None
) -> List[SearchResult]:
    """Perform comprehensive hybrid search across documents and images"""
    
    results = []
    
    # Generate query embeddings
    text_embedding = None
    visual_embedding = None
    
    if query_text:
        text_embedding = generate_text_embedding(query_text)
    
    if query_image_s3_key:
        query_image = load_image_from_s3(query_image_s3_key)
        if query_image:
            visual_embedding = generate_visual_embedding(query_image)
    
    # Calculate adaptive weights
    text_weight, visual_weight = adaptive_weight_calculation(
        query_text, 
        query_image_s3_key is not None
    )
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Search documents
        if "documents" in content_types and text_embedding:
            doc_results = search_documents_hybrid(
                cursor, text_embedding, visual_embedding, 
                text_weight, visual_weight, limit // 2
            )
            results.extend(doc_results)
        
        # Search images
        if "images" in content_types:
            img_results = search_images_hybrid(
                cursor, text_embedding, visual_embedding,
                text_weight, visual_weight, limit // 2, query_text
            )
            results.extend(img_results)
        
        # Sort by hybrid score
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        return results[:limit]
        
    except Exception as e:
        print(f"Error in hybrid search: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def search_documents_hybrid(
    cursor, text_embedding, visual_embedding, 
    text_weight, visual_weight, limit
) -> List[SearchResult]:
    """Search documents with hybrid scoring"""
    
    cursor.execute("""
        SELECT 
            id, title, content, s3_key, department, author,
            CASE 
                WHEN %s IS NOT NULL THEN 1 - (embedding <=> %s)
                ELSE 0
            END as text_similarity,
            0 as visual_similarity  -- Documents don't have visual embeddings
        FROM documents 
        WHERE embedding IS NOT NULL
        ORDER BY 
            CASE 
                WHEN %s IS NOT NULL THEN embedding <=> %s
                ELSE 1
            END
        LIMIT %s;
    """, (text_embedding, text_embedding, text_embedding, text_embedding, limit))
    
    results = []
    for row in cursor.fetchall():
        hybrid_score = calculate_hybrid_score(
            row['text_similarity'], 0, text_weight, visual_weight
        )
        
        results.append(SearchResult(
            id=row['id'],
            type='document',
            title=row['title'],
            content=row['content'][:200] + "..." if row['content'] else "",
            s3_key=row['s3_key'],
            text_similarity=row['text_similarity'],
            visual_similarity=0,
            hybrid_score=hybrid_score,
            metadata={
                'department': row['department'],
                'author': row['author']
            }
        ))
    
    return results

def search_images_hybrid(
    cursor, text_embedding, visual_embedding, 
    text_weight, visual_weight, limit, query_text
) -> List[SearchResult]:
    """Search images with hybrid scoring"""
    
    # Build query based on available embeddings
    if visual_embedding and text_embedding:
        # Full hybrid search
        cursor.execute("""
            SELECT 
                id, filename, description, s3_key, width, height,
                CASE 
                    WHEN %s IS NOT NULL THEN 1 - (visual_embedding <=> %s)
                    ELSE 0
                END as visual_similarity,
                CASE 
                    WHEN description IS NOT NULL AND %s IS NOT NULL 
                    THEN similarity(description, %s)
                    ELSE 0
                END as text_similarity
            FROM images 
            WHERE visual_embedding IS NOT NULL
            ORDER BY 
                (CASE WHEN %s IS NOT NULL THEN visual_embedding <=> %s ELSE 1 END) +
                (CASE WHEN description IS NOT NULL THEN 1 - similarity(description, %s) ELSE 1 END)
            LIMIT %s;
        """, (
            visual_embedding, visual_embedding,
            query_text, query_text,
            visual_embedding, visual_embedding, query_text,
            limit
        ))
    elif visual_embedding:
        # Visual-only search
        cursor.execute("""
            SELECT 
                id, filename, description, s3_key, width, height,
                1 - (visual_embedding <=> %s) as visual_similarity,
                0 as text_similarity
            FROM images 
            WHERE visual_embedding IS NOT NULL
            ORDER BY visual_embedding <=> %s
            LIMIT %s;
        """, (visual_embedding, visual_embedding, limit))
    else:
        # Text-only search on image descriptions
        cursor.execute("""
            SELECT 
                id, filename, description, s3_key, width, height,
                0 as visual_similarity,
                CASE 
                    WHEN description IS NOT NULL 
                    THEN similarity(description, %s)
                    ELSE 0
                END as text_similarity
            FROM images 
            WHERE description IS NOT NULL
            ORDER BY similarity(description, %s) DESC
            LIMIT %s;
        """, (query_text, query_text, limit))
    
    results = []
    for row in cursor.fetchall():
        hybrid_score = calculate_hybrid_score(
            row['text_similarity'], row['visual_similarity'], 
            text_weight, visual_weight
        )
        
        results.append(SearchResult(
            id=row['id'],
            type='image',
            title=row['filename'],
            content=row['description'],
            s3_key=row['s3_key'],
            text_similarity=row['text_similarity'],
            visual_similarity=row['visual_similarity'],
            hybrid_score=hybrid_score,
            metadata={
                'width': row['width'],
                'height': row['height']
            }
        ))
    
    return results
```

### 4. Advanced Hybrid Search SQL Functions
```sql
-- Create function for comprehensive hybrid search
CREATE OR REPLACE FUNCTION hybrid_content_search(
    text_query text DEFAULT NULL,
    text_embedding vector(384) DEFAULT NULL,
    visual_embedding vector(512) DEFAULT NULL,
    text_weight float DEFAULT 0.6,
    visual_weight float DEFAULT 0.4,
    similarity_threshold float DEFAULT 0.3,
    result_limit int DEFAULT 20
)
RETURNS TABLE(
    content_id int,
    content_type text,
    title text,
    description text,
    s3_key text,
    text_sim float,
    visual_sim float,
    hybrid_score float
) AS $$
BEGIN
    RETURN QUERY
    
    -- Search documents
    SELECT 
        d.id as content_id,
        'document'::text as content_type,
        d.title,
        LEFT(d.content, 200) as description,
        d.s3_key,
        CASE 
            WHEN text_embedding IS NOT NULL THEN 1 - (d.embedding <=> text_embedding)
            ELSE 0::float
        END as text_sim,
        0::float as visual_sim,
        CASE 
            WHEN text_embedding IS NOT NULL THEN 
                (1 - (d.embedding <=> text_embedding)) * text_weight
            ELSE 0::float
        END as hybrid_score
    FROM documents d
    WHERE d.embedding IS NOT NULL
    AND (
        text_embedding IS NULL OR 
        (1 - (d.embedding <=> text_embedding)) > similarity_threshold
    )
    
    UNION ALL
    
    -- Search images  
    SELECT 
        i.id as content_id,
        'image'::text as content_type,
        i.filename as title,
        i.description,
        i.s3_key,
        CASE 
            WHEN text_query IS NOT NULL AND i.description IS NOT NULL THEN 
                similarity(i.description, text_query)
            ELSE 0::float
        END as text_sim,
        CASE 
            WHEN visual_embedding IS NOT NULL THEN 1 - (i.visual_embedding <=> visual_embedding)
            ELSE 0::float
        END as visual_sim,
        CASE 
            WHEN text_query IS NOT NULL AND i.description IS NOT NULL AND visual_embedding IS NOT NULL THEN
                (similarity(i.description, text_query) * text_weight) + 
                ((1 - (i.visual_embedding <=> visual_embedding)) * visual_weight)
            WHEN visual_embedding IS NOT NULL THEN
                (1 - (i.visual_embedding <=> visual_embedding)) * visual_weight
            WHEN text_query IS NOT NULL AND i.description IS NOT NULL THEN
                similarity(i.description, text_query) * text_weight
            ELSE 0::float
        END as hybrid_score
    FROM images i
    WHERE i.visual_embedding IS NOT NULL
    AND (
        visual_embedding IS NULL OR 
        (1 - (i.visual_embedding <=> visual_embedding)) > similarity_threshold
    )
    
    ORDER BY hybrid_score DESC
    LIMIT result_limit;
    
END;
$$ LANGUAGE plpgsql;

-- Example usage
SELECT * FROM hybrid_content_search(
    text_query := 'machine learning algorithms',
    text_embedding := (SELECT embedding FROM documents WHERE id = 1),
    visual_embedding := (SELECT visual_embedding FROM images WHERE id = 1),
    text_weight := 0.7,
    visual_weight := 0.3,
    similarity_threshold := 0.4,
    result_limit := 15
);
```

### 5. Performance Optimization
```sql
-- Create composite indexes for hybrid search
CREATE INDEX idx_documents_embedding_content ON documents 
USING gin(to_tsvector('english', content)) 
WHERE embedding IS NOT NULL;

CREATE INDEX idx_images_description_embedding ON images 
USING gin(to_tsvector('english', description)) 
WHERE visual_embedding IS NOT NULL;

-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM hybrid_content_search(
    'database optimization',
    NULL,
    NULL,
    0.6,
    0.4,
    0.3,
    10
);
```

## Testing Hybrid Search
```python
# Example test cases
def test_hybrid_search():
    """Test different hybrid search scenarios"""
    
    # Test 1: Text-only search
    results = hybrid_search(
        query_text="machine learning algorithms",
        search_mode=SearchMode.TEXT_ONLY,
        content_types=["documents", "images"]
    )
    print(f"Text-only search returned {len(results)} results")
    
    # Test 2: Visual-only search
    results = hybrid_search(
        query_text="",
        query_image_s3_key="images/sample-chart.jpg",
        search_mode=SearchMode.VISUAL_ONLY,
        content_types=["images"]
    )
    print(f"Visual-only search returned {len(results)} results")
    
    # Test 3: Hybrid search
    results = hybrid_search(
        query_text="data visualization charts",
        query_image_s3_key="images/sample-chart.jpg",
        search_mode=SearchMode.HYBRID,
        content_types=["documents", "images"]
    )
    print(f"Hybrid search returned {len(results)} results")
    
    # Print top results
    for i, result in enumerate(results[:5]):
        print(f"\nResult {i+1}:")
        print(f"  Type: {result.type}")
        print(f"  Title: {result.title}")
        print(f"  Hybrid Score: {result.hybrid_score:.3f}")
        print(f"  Text Similarity: {result.text_similarity:.3f}")
        print(f"  Visual Similarity: {result.visual_similarity:.3f}")

if __name__ == "__main__":
    test_hybrid_search()
```

## Next Steps
Continue to Module 05 to implement vector indexing and performance optimization strategies.
