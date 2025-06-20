-- Database Migration Script for pgvector-hybrid-search-demo-py
-- Generated: 2025-06-20T02:54:42.381Z
-- Target: Aurora PostgreSQL Aurora PostgreSQL 16
-- Region: us-east-1

BEGIN;

-- Migration tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(255) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT NOW(),
    description TEXT
);

-- Insert migration record
INSERT INTO schema_migrations (version, description) 
VALUES ('001_initial_schema', 'pgvector-hybrid-search-demo-py initial database schema')
ON CONFLICT (version) DO NOTHING;

-- Enable required extensions


-- pgvector for vector operations
CREATE EXTENSION IF NOT EXISTS vector;




-- Create schemas for organization
CREATE SCHEMA IF NOT EXISTS app_data;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS logs;




-- VECTOR SEARCH TABLES
-- Documents table with vector embeddings
CREATE TABLE IF NOT EXISTS app_data.documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384), -- Standard sentence-transformer dimension
    document_type VARCHAR(50) DEFAULT 'text',
    source_url TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Images table for multi-modal search
CREATE TABLE IF NOT EXISTS app_data.images (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    description TEXT,
    image_embedding vector(512), -- CLIP model dimension
    text_embedding vector(384),  -- For cross-modal search
    file_path TEXT,
    file_size INTEGER,
    dimensions JSONB, -- {width: 1920, height: 1080}
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector search indexes using HNSW for optimal performance
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON app_data.documents 
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_images_image_embedding ON app_data.images 
    USING hnsw (image_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_images_text_embedding ON app_data.images 
    USING hnsw (text_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Additional indexes for hybrid search
CREATE INDEX IF NOT EXISTS idx_documents_content_fts ON app_data.documents USING GIN (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_documents_type ON app_data.documents (document_type);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON app_data.documents USING GIN (metadata);

-- Vector search functions
CREATE OR REPLACE FUNCTION app_data.semantic_search(
    query_embedding vector(384),
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 10
) RETURNS TABLE (
    id INTEGER,
    title TEXT,
    content TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        CASE 
            WHEN LENGTH(d.content) > 200 THEN LEFT(d.content, 200) || '...'
            ELSE d.content
        END as content,
        1 - (d.embedding <=> query_embedding) as similarity
    FROM app_data.documents d
    WHERE d.embedding IS NOT NULL
      AND 1 - (d.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql STABLE;

-- Hybrid search combining vector and full-text search
CREATE OR REPLACE FUNCTION app_data.hybrid_search(
    query_text TEXT,
    query_embedding vector(384),
    vector_weight FLOAT DEFAULT 0.7,
    text_weight FLOAT DEFAULT 0.3,
    max_results INTEGER DEFAULT 10
) RETURNS TABLE (
    id INTEGER,
    title TEXT,
    content TEXT,
    combined_score FLOAT,
    vector_score FLOAT,
    text_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            d.id,
            d.title,
            d.content,
            1 - (d.embedding <=> query_embedding) as vector_score
        FROM app_data.documents d
        WHERE d.embedding IS NOT NULL
    ),
    text_results AS (
        SELECT 
            d.id,
            ts_rank_cd(to_tsvector('english', d.content), plainto_tsquery('english', query_text)) as text_score
        FROM app_data.documents d
        WHERE to_tsvector('english', d.content) @@ plainto_tsquery('english', query_text)
    )
    SELECT 
        v.id,
        v.title,
        CASE 
            WHEN LENGTH(v.content) > 200 THEN LEFT(v.content, 200) || '...'
            ELSE v.content
        END as content,
        (COALESCE(v.vector_score, 0) * vector_weight + COALESCE(t.text_score, 0) * text_weight) as combined_score,
        v.vector_score,
        COALESCE(t.text_score, 0) as text_score
    FROM vector_results v
    LEFT JOIN text_results t ON v.id = t.id
    WHERE v.vector_score > 0.5 OR t.text_score > 0.1
    ORDER BY combined_score DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql STABLE;

-- Comprehensive Vector Search Sample Data
-- This includes diverse document types for realistic semantic search demonstrations

-- Technical Documentation
INSERT INTO app_data.documents (title, content, document_type, source_url, metadata) VALUES
('PostgreSQL Vector Extensions Guide', 'pgvector provides vector similarity search capabilities for PostgreSQL databases, enabling semantic search and AI applications. The extension supports exact and approximate nearest neighbor search using L2 distance, inner product, and cosine distance. Key features include: HNSW indexing for fast approximate search, exact brute-force search for smaller datasets, support for up to 16,000 dimensions, and integration with popular machine learning frameworks like Python and JavaScript.', 'technical', 'https://github.com/pgvector/pgvector', '{"tags": ["postgresql", "vector", "machine_learning"], "difficulty": "intermediate", "reading_time": "5 minutes"}'),

('Database Performance Optimization', 'Learn advanced techniques for optimizing PostgreSQL performance including indexing strategies, query optimization, and configuration tuning. Performance optimization involves understanding query execution plans, choosing appropriate index types (B-tree, Hash, GiST, GIN, BRIN), configuring memory settings like shared_buffers and work_mem, and implementing connection pooling. Regular maintenance tasks include VACUUM, ANALYZE, and REINDEX operations. Monitoring tools like pg_stat_statements help identify slow queries and bottlenecks.', 'guide', 'https://www.postgresql.org/docs/current/performance-tips.html', '{"tags": ["performance", "optimization", "indexing"], "difficulty": "advanced", "reading_time": "15 minutes"}'),

('Machine Learning Pipeline Integration', 'Integrate machine learning workflows directly with your PostgreSQL database using extensions like pgvector for embedding storage and similarity search. Modern ML pipelines can leverage PostgreSQL for feature stores, model versioning, and real-time inference. Popular frameworks like TensorFlow, PyTorch, and scikit-learn can connect directly to PostgreSQL for training data access and model deployment. Vector embeddings from language models, image encoders, and recommendation systems can be stored and queried efficiently using pgvector.', 'tutorial', 'https://mlops.community/postgresql-ml-pipeline/', '{"tags": ["machine_learning", "embeddings", "pipeline"], "difficulty": "advanced", "reading_time": "20 minutes"}'),

('Aurora PostgreSQL Architecture', 'Amazon Aurora PostgreSQL-Compatible Edition provides enhanced performance, availability, and durability for cloud-native applications. Aurora separates compute and storage layers, enabling rapid scaling and fault tolerance. Key architectural benefits include: automatic failover in under 30 seconds, up to 15 read replicas, continuous backup to S3, point-in-time recovery, and cross-region automated backups. Aurora Serverless provides on-demand, auto-scaling configurations for variable workloads.', 'documentation', 'https://docs.aws.amazon.com/AmazonRDS/latest/AuroraUserGuide/', '{"tags": ["aurora", "aws", "cloud", "architecture"], "difficulty": "intermediate", "reading_time": "10 minutes"}'),

('Real-time Analytics Pipeline Design', 'Build scalable real-time analytics using PostgreSQL with time-series data, materialized views, and continuous aggregation patterns. Real-time analytics requires efficient data ingestion, storage optimization, and query performance. Techniques include partitioning large tables by time, using materialized views for pre-computed aggregations, implementing streaming data processing with tools like Apache Kafka, and leveraging PostgreSQL extensions like TimescaleDB for time-series workloads.', 'architecture', 'https://www.timescale.com/blog/real-time-analytics/', '{"tags": ["analytics", "real_time", "time_series"], "difficulty": "advanced", "reading_time": "12 minutes"}'),

-- Product Documentation
('API Authentication Guide', 'Secure your REST APIs using JWT tokens, OAuth 2.0, and API key authentication. Implement rate limiting, input validation, and proper error handling to prevent common security vulnerabilities. Best practices include using HTTPS, implementing proper CORS policies, logging security events, and regular security audits. Consider using API gateways for centralized authentication and monitoring.', 'documentation', 'https://api-docs.example.com/auth', '{"tags": ["api", "security", "authentication"], "difficulty": "intermediate", "reading_time": "8 minutes"}'),

('Microservices Communication Patterns', 'Design effective communication between microservices using synchronous and asynchronous patterns. Synchronous communication includes REST APIs and GraphQL, while asynchronous patterns use message queues, event streaming, and publish-subscribe models. Service mesh technologies like Istio provide traffic management, security, and observability for microservice architectures.', 'architecture', 'https://microservices.io/patterns/', '{"tags": ["microservices", "communication", "architecture"], "difficulty": "advanced", "reading_time": "18 minutes"}'),

-- Business Content
('Digital Transformation Strategy', 'Enterprise digital transformation requires strategic planning, technology adoption, and cultural change management. Key success factors include executive leadership, cross-functional collaboration, agile methodologies, and data-driven decision making. Organizations must balance innovation with operational stability while building digital capabilities and modernizing legacy systems.', 'business', 'https://enterprise.example.com/transformation', '{"tags": ["digital_transformation", "strategy", "enterprise"], "difficulty": "beginner", "reading_time": "6 minutes"}'),

('Cloud Migration Best Practices', 'Successfully migrate applications to the cloud using the 6 Rs framework: Rehost, Replatform, Refactor, Repurchase, Retain, and Retire. Assessment and planning phases should evaluate technical dependencies, security requirements, and cost implications. Migration strategies include lift-and-shift for quick wins, followed by optimization and modernization phases.', 'guide', 'https://aws.amazon.com/cloud-migration/', '{"tags": ["cloud", "migration", "aws"], "difficulty": "intermediate", "reading_time": "14 minutes"}'),

-- Academic Research
('Distributed Systems Consensus Algorithms', 'Consensus algorithms like Raft and PBFT enable distributed systems to agree on shared state despite node failures and network partitions. These algorithms are fundamental to blockchain technologies, distributed databases, and fault-tolerant systems. Key properties include safety (consistency) and liveness (progress) guarantees under various failure scenarios.', 'research', 'https://raft.github.io/', '{"tags": ["distributed_systems", "consensus", "algorithms"], "difficulty": "expert", "reading_time": "25 minutes"}'),

('Natural Language Processing Advances', 'Recent advances in NLP include transformer architectures, attention mechanisms, and large language models like GPT and BERT. These models demonstrate emergent capabilities in few-shot learning, reasoning, and multimodal understanding. Applications span machine translation, text summarization, question answering, and code generation.', 'research', 'https://arxiv.org/abs/1706.03762', '{"tags": ["nlp", "transformers", "machine_learning"], "difficulty": "expert", "reading_time": "30 minutes"}'),

-- User Guides and Tutorials
('Getting Started with Docker', 'Docker containerization simplifies application deployment and development workflows. Learn to create Dockerfiles, manage container lifecycles, use Docker Compose for multi-service applications, and implement best practices for security and performance. Container registries enable sharing and versioning of container images across teams.', 'tutorial', 'https://docs.docker.com/get-started/', '{"tags": ["docker", "containers", "devops"], "difficulty": "beginner", "reading_time": "12 minutes"}'),

('Kubernetes Deployment Strategies', 'Deploy applications on Kubernetes using various strategies including blue-green deployments, canary releases, and rolling updates. Kubernetes provides declarative configuration, automatic scaling, service discovery, and fault tolerance. Key concepts include pods, services, ingress controllers, and persistent volumes for stateful applications.', 'tutorial', 'https://kubernetes.io/docs/concepts/', '{"tags": ["kubernetes", "deployment", "orchestration"], "difficulty": "intermediate", "reading_time": "20 minutes"}'),

-- News and Updates
('Tech Industry Trends 2024', 'Emerging technology trends include artificial intelligence integration, edge computing expansion, quantum computing advances, and sustainable technology practices. Organizations are investing in AI-powered automation, real-time data processing at the edge, and carbon-neutral computing infrastructure. Regulatory compliance and ethical AI considerations drive technology adoption decisions.', 'news', 'https://techtrends.example.com/2024', '{"tags": ["trends", "technology", "2024"], "difficulty": "beginner", "reading_time": "7 minutes"}'),

('Open Source Software Adoption', 'Enterprise adoption of open source software continues accelerating, driven by cost savings, innovation speed, and vendor independence. Popular categories include cloud-native technologies, data analytics platforms, and machine learning frameworks. Organizations must consider support models, security practices, and license compliance when adopting open source solutions.', 'news', 'https://opensource.example.com/adoption', '{"tags": ["open_source", "enterprise", "adoption"], "difficulty": "beginner", "reading_time": "5 minutes"}')

ON CONFLICT DO NOTHING;

-- Sample image data for multi-modal search
INSERT INTO app_data.images (filename, description, file_path, file_size, dimensions, metadata) VALUES
('architecture_diagram_1.png', 'Microservices architecture diagram showing API gateway, service mesh, and database connections', '/images/diagrams/architecture_diagram_1.png', 245760, '{"width": 1920, "height": 1080}', '{"tags": ["architecture", "microservices", "diagram"], "created_by": "engineering_team", "format": "png"}'),
('database_schema_2.png', 'PostgreSQL database schema visualization with tables, relationships, and indexes', '/images/schemas/database_schema_2.png', 189440, '{"width": 1600, "height": 1200}', '{"tags": ["database", "schema", "postgresql"], "created_by": "data_team", "format": "png"}'),
('dashboard_mockup_3.png', 'Analytics dashboard mockup with charts, graphs, and key performance indicators', '/images/mockups/dashboard_mockup_3.png', 312320, '{"width": 2560, "height": 1440}', '{"tags": ["dashboard", "analytics", "ui"], "created_by": "design_team", "format": "png"}'),
('cloud_infrastructure_4.png', 'AWS cloud infrastructure diagram with VPC, subnets, load balancers, and auto-scaling groups', '/images/cloud/infrastructure_4.png', 278540, '{"width": 2048, "height": 1536}', '{"tags": ["aws", "cloud", "infrastructure"], "created_by": "devops_team", "format": "png"}'),
('ml_pipeline_5.png', 'Machine learning pipeline flowchart from data ingestion to model deployment', '/images/ml/pipeline_5.png', 201680, '{"width": 1800, "height": 1200}', '{"tags": ["machine_learning", "pipeline", "workflow"], "created_by": "ml_team", "format": "png"}')

ON CONFLICT DO NOTHING;

-- Generate additional sample documents using stored procedure
CREATE OR REPLACE FUNCTION app_data.generate_sample_documents(num_documents INTEGER DEFAULT 50)
RETURNS void AS $$
DECLARE
    i INTEGER;
    doc_types TEXT[] := ARRAY['technical', 'guide', 'tutorial', 'documentation', 'research', 'business', 'news'];
    sample_titles TEXT[] := ARRAY[
        'Advanced PostgreSQL Techniques', 'Cloud Security Best Practices', 'API Design Principles',
        'Data Visualization Methods', 'Software Testing Strategies', 'DevOps Automation',
        'Scalable System Architecture', 'Modern Web Development', 'Mobile App Performance',
        'Database Optimization Tips', 'Cybersecurity Frameworks', 'AI Ethics Guidelines'
    ];
    sample_content TEXT[] := ARRAY[
        'This comprehensive guide covers advanced database management techniques and optimization strategies.',
        'Learn essential security practices for cloud-based applications and infrastructure.',
        'Design robust and maintainable APIs following industry best practices and standards.',
        'Explore effective methods for presenting data insights through charts and interactive visualizations.',
        'Implement comprehensive testing strategies including unit, integration, and end-to-end testing.',
        'Automate development workflows using CI/CD pipelines and infrastructure as code.',
        'Build scalable systems that handle growth in users, data, and transaction volume.',
        'Master modern web development frameworks, tools, and deployment strategies.',
        'Optimize mobile applications for performance, battery life, and user experience.',
        'Advanced techniques for database performance tuning and query optimization.',
        'Comprehensive cybersecurity frameworks for enterprise risk management.',
        'Ethical considerations and best practices for AI system development and deployment.'
    ];
BEGIN
    FOR i IN 1..num_documents LOOP
        INSERT INTO app_data.documents (
            title,
            content,
            document_type,
            metadata
        ) VALUES (
            sample_titles[(i % array_length(sample_titles, 1)) + 1] || ' - Part ' || i,
            sample_content[(i % array_length(sample_content, 1)) + 1] || ' This document provides detailed information and practical examples for implementation.',
            doc_types[(i % array_length(doc_types, 1)) + 1],
            jsonb_build_object(
                'tags', ARRAY['generated', 'sample', 'demo'],
                'difficulty', CASE (RANDOM() * 3)::INTEGER 
                    WHEN 0 THEN 'beginner'
                    WHEN 1 THEN 'intermediate'
                    ELSE 'advanced'
                END,
                'reading_time', (RANDOM() * 20 + 5)::INTEGER || ' minutes'
            )
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Execute to create additional sample documents
SELECT app_data.generate_sample_documents(50);








-- System logging table
CREATE TABLE IF NOT EXISTS logs.system_events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    description TEXT,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_system_events_type_time ON logs.system_events (event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_system_events_time ON logs.system_events (created_at DESC);

-- Performance monitoring view
CREATE OR REPLACE VIEW analytics.database_performance AS
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation,
    most_common_vals,
    most_common_freqs
FROM pg_stats 
WHERE schemaname IN ('app_data', 'analytics')
ORDER BY schemaname, tablename, attname;

-- Database size monitoring function
CREATE OR REPLACE FUNCTION analytics.get_database_size_info()
RETURNS TABLE (
    database_name TEXT,
    size_pretty TEXT,
    size_bytes BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        current_database()::TEXT as database_name,
        pg_size_pretty(pg_database_size(current_database())) as size_pretty,
        pg_database_size(current_database()) as size_bytes;
END;
$$ LANGUAGE plpgsql STABLE;

-- Update table statistics
ANALYZE;

-- Log successful migration
INSERT INTO logs.system_events (event_type, description, details)
VALUES (
    'schema_migration',
    'Initial schema migration completed successfully',
    jsonb_build_object(
        'migration_version', '001_initial_schema',
        'repository_name', 'pgvector-hybrid-search-demo-py',
        'database_version', 'Aurora PostgreSQL 16',
        'use_cases', 'Vector/Hybrid Search',
        'region', 'us-east-1'
    )
);

COMMIT;

-- Post-migration verification
\echo 'Migration completed successfully!'
\echo 'Verifying extensions:'
SELECT name, default_version, installed_version FROM pg_available_extensions WHERE installed_version IS NOT NULL;

\echo 'Verifying tables:'
SELECT schemaname, tablename, tableowner FROM pg_tables WHERE schemaname IN ('app_data', 'analytics', 'logs') ORDER BY schemaname, tablename;

\echo 'Database ready for pgvector-hybrid-search-demo-py!'
