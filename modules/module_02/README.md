# Module 02: Document Processing & Text Embeddings

## Overview
Implementing document processing pipeline with text embedding generation for Aurora PostgreSQL vector search.

## Text Embedding Pipeline

### 1. Document Processing Setup
```python
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
import boto3
import PyPDF2
import io
import os
from typing import List, Optional

# Initialize embedding model
text_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_db_connection():
    """Create Aurora PostgreSQL connection"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT', 5432),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
```

### 2. Text Extraction Functions
```python
def extract_text_from_pdf(s3_key: str) -> str:
    """Extract text from PDF stored in S3"""
    s3_client = boto3.client('s3')
    
    try:
        # Download PDF from S3
        response = s3_client.get_object(
            Bucket=os.getenv('S3_BUCKET'), 
            Key=s3_key
        )
        pdf_content = response['Body'].read()
        
        # Extract text using PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text_content = ""
        
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        return text_content.strip()
    
    except Exception as e:
        print(f"Error extracting text from {s3_key}: {e}")
        return ""

def process_document_content(content: str) -> List[str]:
    """Split document into chunks for better embedding quality"""
    # Split into sentences/paragraphs (adjust based on your needs)
    chunks = []
    sentences = content.split('. ')
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 500:  # Keep chunks under 500 chars
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
```

### 3. Embedding Generation
```python
def generate_text_embedding(text: str) -> List[float]:
    """Generate embedding for text using sentence transformer"""
    try:
        # Generate embedding
        embedding = text_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def store_document_with_embedding(
    title: str,
    content: str,
    s3_key: str = None,
    department: str = None,
    author: str = None,
    tags: List[str] = None
):
    """Store document with generated embedding in Aurora PostgreSQL"""
    
    # Generate embedding for full content
    embedding = generate_text_embedding(content)
    if not embedding:
        print("Failed to generate embedding")
        return None
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO documents 
            (title, content, s3_key, embedding, department, author, tags)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            title, content, s3_key, embedding, 
            department, author, tags
        ))
        
        document_id = cursor.fetchone()[0]
        conn.commit()
        
        print(f"Document stored with ID: {document_id}")
        return document_id
        
    except Exception as e:
        print(f"Error storing document: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()
```

### 4. Batch Processing from S3
```python
def process_s3_documents(bucket_name: str, prefix: str = ""):
    """Process all documents in S3 bucket/prefix"""
    s3_client = boto3.client('s3')
    
    try:
        # List objects in S3
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            print("No objects found")
            return
        
        for obj in response['Contents']:
            s3_key = obj['Key']
            
            # Skip non-document files
            if not s3_key.endswith(('.pdf', '.txt', '.docx')):
                continue
            
            print(f"Processing: {s3_key}")
            
            # Extract text based on file type
            if s3_key.endswith('.pdf'):
                content = extract_text_from_pdf(s3_key)
            elif s3_key.endswith('.txt'):
                # Handle text files
                response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                content = response['Body'].read().decode('utf-8')
            
            if content and len(content.strip()) > 10:
                # Extract metadata from S3 key
                title = os.path.basename(s3_key)
                department = s3_key.split('/')[0] if '/' in s3_key else 'General'
                
                # Store document
                store_document_with_embedding(
                    title=title,
                    content=content,
                    s3_key=s3_key,
                    department=department
                )
    
    except Exception as e:
        print(f"Error processing S3 documents: {e}")

# Example usage
if __name__ == "__main__":
    # Process documents from S3
    process_s3_documents('your-document-bucket', 'documents/')
    
    print("Document processing completed!")
```

## Semantic Search Implementation
```python
def semantic_search(query: str, limit: int = 10) -> List[dict]:
    """Perform semantic search on documents"""
    
    # Generate query embedding
    query_embedding = generate_text_embedding(query)
    if not query_embedding:
        return []
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                id, title, content, s3_key, department, author,
                1 - (embedding <=> %s) AS similarity
            FROM documents 
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s
            LIMIT %s;
        """, (query_embedding, query_embedding, limit))
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
        
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# Test semantic search
results = semantic_search("database performance optimization")
for result in results:
    print(f"Title: {result['title']}")
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Department: {result['department']}")
    print("---")
```

## Next Steps
Continue to Module 03 to implement image processing and visual embeddings.
