# Module 03: Image Processing & Visual Embeddings

## Overview
Implementing image processing pipeline with visual embedding generation for multi-modal search capabilities in Aurora PostgreSQL.

## Visual Embedding Pipeline

### 1. Image Processing Setup
```python
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import boto3
import io
import numpy as np
from typing import List, Tuple
import psycopg2

# Initialize ResNet model for visual embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet_model = resnet50(pretrained=True)
resnet_model.fc = torch.nn.Identity()  # Remove final classification layer
resnet_model.eval()
resnet_model.to(device)

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

### 2. Image Processing Functions
```python
def load_image_from_s3(s3_key: str) -> Image.Image:
    """Load image from S3 bucket"""
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.get_object(
            Bucket=os.getenv('S3_BUCKET'),
            Key=s3_key
        )
        image_content = response['Body'].read()
        image = Image.open(io.BytesIO(image_content)).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image from S3: {e}")
        return None

def generate_visual_embedding(image: Image.Image) -> List[float]:
    """Generate visual embedding using ResNet50"""
    try:
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = resnet_model(image_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating visual embedding: {e}")
        return None

def extract_image_metadata(image: Image.Image, s3_key: str) -> dict:
    """Extract metadata from image"""
    return {
        'width': image.width,
        'height': image.height,
        'format': image.format,
        'mode': image.mode,
        'filename': s3_key.split('/')[-1]
    }
```

### 3. Image Storage with Embeddings
```python
def store_image_with_embedding(
    s3_key: str,
    description: str = None,
    tags: List[str] = None
) -> int:
    """Store image metadata and visual embedding in Aurora PostgreSQL"""
    
    # Load and process image
    image = load_image_from_s3(s3_key)
    if not image:
        print(f"Failed to load image: {s3_key}")
        return None
    
    # Generate visual embedding
    visual_embedding = generate_visual_embedding(image)
    if not visual_embedding:
        print(f"Failed to generate embedding for: {s3_key}")
        return None
    
    # Extract metadata
    metadata = extract_image_metadata(image, s3_key)
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO images 
            (filename, description, s3_key, visual_embedding, width, height, 
             file_size, content_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            metadata['filename'],
            description,
            s3_key,
            visual_embedding,
            metadata['width'],
            metadata['height'],
            0,  # File size can be calculated from S3 metadata
            f"image/{metadata['format'].lower()}" if metadata['format'] else 'image/jpeg'
        ))
        
        image_id = cursor.fetchone()[0]
        conn.commit()
        
        print(f"Image stored with ID: {image_id}")
        return image_id
        
    except Exception as e:
        print(f"Error storing image: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()

def process_s3_images(bucket_name: str, prefix: str = "images/"):
    """Process all images in S3 bucket/prefix"""
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            print("No images found")
            return
        
        for obj in response['Contents']:
            s3_key = obj['Key']
            
            # Skip non-image files
            if not s3_key.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                continue
            
            print(f"Processing image: {s3_key}")
            
            # Extract description from filename or S3 metadata
            filename = s3_key.split('/')[-1]
            description = filename.split('.')[0].replace('_', ' ').replace('-', ' ')
            
            # Store image with embedding
            store_image_with_embedding(
                s3_key=s3_key,
                description=description
            )
    
    except Exception as e:
        print(f"Error processing S3 images: {e}")
```

### 4. Visual Similarity Search
```python
def visual_similarity_search(query_s3_key: str, limit: int = 10) -> List[dict]:
    """Find visually similar images"""
    
    # Load and process query image
    query_image = load_image_from_s3(query_s3_key)
    if not query_image:
        return []
    
    # Generate query embedding
    query_embedding = generate_visual_embedding(query_image)
    if not query_embedding:
        return []
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                id, filename, description, s3_key, width, height,
                1 - (visual_embedding <=> %s) AS similarity
            FROM images 
            WHERE visual_embedding IS NOT NULL
            AND s3_key != %s  -- Exclude the query image itself
            ORDER BY visual_embedding <=> %s
            LIMIT %s;
        """, (query_embedding, query_s3_key, query_embedding, limit))
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
        
    except Exception as e:
        print(f"Error in visual similarity search: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

def cross_modal_search(text_query: str, limit: int = 10) -> List[dict]:
    """Find images based on text description"""
    from sentence_transformers import SentenceTransformer, util
    
    # Load CLIP model for cross-modal search
    clip_model = SentenceTransformer('clip-ViT-B-32')
    
    # Generate text embedding
    text_embedding = clip_model.encode(text_query)
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # For this example, we'll search by description similarity
        # In production, you'd want to use CLIP embeddings for both text and images
        cursor.execute("""
            SELECT 
                id, filename, description, s3_key, width, height,
                similarity(description, %s) AS text_similarity
            FROM images 
            WHERE description IS NOT NULL
            ORDER BY similarity(description, %s) DESC
            LIMIT %s;
        """, (text_query, text_query, limit))
        
        results = cursor.fetchall()
        return [dict(row) for row in results]
        
    except Exception as e:
        print(f"Error in cross-modal search: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# Example usage
if __name__ == "__main__":
    # Process images from S3
    process_s3_images('your-image-bucket', 'images/')
    
    # Test visual similarity
    similar_images = visual_similarity_search('images/sample.jpg')
    for img in similar_images:
        print(f"Similar image: {img['filename']}")
        print(f"Similarity: {img['similarity']:.3f}")
        print("---")
    
    print("Image processing completed!")
```

## Advanced Visual Features
```sql
-- Create function for image similarity threshold filtering
CREATE OR REPLACE FUNCTION find_similar_images(
    query_embedding vector(512),
    similarity_threshold float DEFAULT 0.8,
    result_limit int DEFAULT 10
)
RETURNS TABLE(
    image_id int,
    filename text,
    s3_key text,
    similarity_score float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        i.id,
        i.filename,
        i.s3_key,
        (1 - (i.visual_embedding <=> query_embedding)) as similarity_score
    FROM images i
    WHERE i.visual_embedding IS NOT NULL
    AND (1 - (i.visual_embedding <=> query_embedding)) > similarity_threshold
    ORDER BY i.visual_embedding <=> query_embedding
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Test the function
SELECT * FROM find_similar_images(
    (SELECT visual_embedding FROM images WHERE id = 1),
    0.7,
    5
);
```

## Next Steps
Continue to Module 04 to implement hybrid search combining text and visual similarity.
