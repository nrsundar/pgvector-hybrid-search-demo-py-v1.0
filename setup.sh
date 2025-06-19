#!/bin/bash
# Database Setup Script for pgvector-hybrid-search-demo-py
# This script connects to Aurora PostgreSQL and sets up PostGIS

set -e

echo "Setting up pgvector-hybrid-search-demo-py on Aurora PostgreSQL..."

# Database connection variables (update with actual CloudFormation outputs)
DB_HOST="${DB_HOST:-<database-endpoint-from-cloudformation>}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-postgres}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-SecurePassword123!}"

CONNECTION_STRING="postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"

echo "Connecting to database at $DB_HOST..."

# Enable PostGIS extension
echo "Enabling PostGIS extension..."
psql "$CONNECTION_STRING" -c "CREATE EXTENSION IF NOT EXISTS postgis;"
psql "$CONNECTION_STRING" -c "CREATE EXTENSION IF NOT EXISTS postgis_topology;"

# Create sample spatial tables for general use cases
echo "Creating sample spatial tables..."
psql "$CONNECTION_STRING" << 'EOF'
-- Create locations table with spatial data
CREATE TABLE IF NOT EXISTS locations (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    coordinates GEOMETRY(POINT, 4326),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create spatial index
CREATE INDEX IF NOT EXISTS idx_locations_coordinates ON locations USING GIST (coordinates);

-- Insert sample data
INSERT INTO locations (name, description, coordinates) VALUES
('Sample Point 1', 'First sample location', ST_GeomFromText('POINT(-122.4194 37.7749)', 4326)),
('Sample Point 2', 'Second sample location', ST_GeomFromText('POINT(-74.0060 40.7128)', 4326))
ON CONFLICT DO NOTHING;
EOF


echo "Verifying PostGIS installation..."
psql "$CONNECTION_STRING" -c "SELECT PostGIS_Version();"

echo "Database setup completed successfully!"
echo "Connection string: $CONNECTION_STRING"
