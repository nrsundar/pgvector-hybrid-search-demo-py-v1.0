# Module 08: Embedding Models & Fine-tuning

## Overview
Implementing custom embedding models, fine-tuning strategies, and domain-specific similarity functions for optimized pgvector performance in Aurora PostgreSQL.

## Custom Embedding Models

### 1. Fine-Tuning Text Embeddings
```python
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple

class CustomEmbeddingTrainer:
    def __init__(self, base_model: str = 'all-MiniLM-L6-v2'):
        self.base_model = SentenceTransformer(base_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model.to(self.device)
    
    def prepare_training_data(
        self, 
        positive_pairs: List[Tuple[str, str]],
        negative_pairs: List[Tuple[str, str]] = None
    ) -> List[InputExample]:
        """Prepare training data for fine-tuning"""
        
        train_examples = []
        
        # Add positive pairs (similar documents)
        for text1, text2 in positive_pairs:
            train_examples.append(InputExample(texts=[text1, text2], label=1.0))
        
        # Add negative pairs (dissimilar documents)
        if negative_pairs:
            for text1, text2 in negative_pairs:
                train_examples.append(InputExample(texts=[text1, text2], label=0.0))
        
        return train_examples
    
    def fine_tune_model(
        self,
        train_examples: List[InputExample],
        output_path: str = './fine_tuned_model',
        epochs: int = 4,
        warmup_steps: int = 100
    ):
        """Fine-tune the embedding model for domain-specific data"""
        
        # Create DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # Define loss function (Cosine Similarity Loss)
        train_loss = losses.CosineSimilarityLoss(self.base_model)
        
        # Fine-tune the model
        self.base_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True
        )
        
        print(f"Fine-tuned model saved to: {output_path}")
```

### 2. Custom Similarity Functions
```sql
-- Create custom similarity functions for domain-specific use cases
CREATE OR REPLACE FUNCTION weighted_cosine_similarity(
    embedding1 vector(384),
    embedding2 vector(384),
    feature_weights vector(384) DEFAULT NULL
)
RETURNS float AS $$
DECLARE
    weighted_emb1 vector(384);
    weighted_emb2 vector(384);
    cosine_sim float;
BEGIN
    -- Apply feature weights if provided
    IF feature_weights IS NOT NULL THEN
        weighted_emb1 := embedding1 * feature_weights;
        weighted_emb2 := embedding2 * feature_weights;
    ELSE
        weighted_emb1 := embedding1;
        weighted_emb2 := embedding2;
    END IF;
    
    -- Calculate weighted cosine similarity
    cosine_sim := 1 - (weighted_emb1 <=> weighted_emb2);
    
    RETURN cosine_sim;
END;
$$ LANGUAGE plpgsql;

-- Create function for domain-aware similarity
CREATE OR REPLACE FUNCTION domain_aware_similarity(
    embedding1 vector(384),
    embedding2 vector(384),
    domain1 text,
    domain2 text,
    content_type1 text DEFAULT 'document',
    content_type2 text DEFAULT 'document'
)
RETURNS float AS $$
DECLARE
    base_similarity float;
    domain_boost float := 1.0;
    type_boost float := 1.0;
    final_similarity float;
BEGIN
    -- Calculate base cosine similarity
    base_similarity := 1 - (embedding1 <=> embedding2);
    
    -- Apply domain boost
    IF domain1 = domain2 THEN
        domain_boost := 1.2;  -- 20% boost for same domain
    ELSIF domain1 IN ('Engineering', 'Technology') AND domain2 IN ('Engineering', 'Technology') THEN
        domain_boost := 1.1;  -- 10% boost for related domains
    END IF;
    
    -- Apply content type boost
    IF content_type1 = content_type2 THEN
        type_boost := 1.1;  -- 10% boost for same content type
    END IF;
    
    -- Calculate final similarity with boosts
    final_similarity := base_similarity * domain_boost * type_boost;
    
    -- Cap at 1.0
    RETURN LEAST(final_similarity, 1.0);
END;
$$ LANGUAGE plpgsql;
```

## Next Steps
Continue to Module 09 to implement production scaling and optimization strategies.
