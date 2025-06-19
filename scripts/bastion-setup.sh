#!/bin/bash
set -e

echo "🚀 Setting up pgvector-hybrid-search-demo-py on Ubuntu Bastion Host for Aurora PostgreSQL connectivity..."

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install PostgreSQL client and essential tools (NO SERVER INSTALLATION)
sudo apt install -y wget ca-certificates
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" | sudo tee /etc/apt/sources.list.d/pgdg.list
sudo apt update

# Install PostgreSQL client tools only
sudo apt install -y postgresql-client-15.4 postgresql-contrib postgresql-client-common

# Install development tools
sudo apt install -y build-essential git curl unzip


# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv python3-dev libpq-dev
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install psycopg2-binary boto3 flask sqlalchemy


# Set up environment variables for Aurora PostgreSQL connection
echo "Setting up database connection environment..."
cat > /home/ubuntu/.env << 'EOF'
# Aurora PostgreSQL Connection Details
# Update these values with actual CloudFormation outputs
DB_HOST=<database-endpoint-from-cloudformation>
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=SecurePassword123!
DATABASE_URL=postgresql://postgres:SecurePassword123!@<database-endpoint>:5432/postgres
AWS_REGION=us-east-1
EOF

# Create database connection test script
cat > /home/ubuntu/test-db-connection.sh << 'EOF'
#!/bin/bash
source /home/ubuntu/.env

echo "Testing connection to Aurora PostgreSQL..."
echo "Database Host: $DB_HOST"
echo "Database Port: $DB_PORT"

# Test basic connection
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" -c "SELECT version();"

if [ $? -eq 0 ]; then
    echo "✅ Database connection successful!"
    
    # Test PostGIS extension
    echo "Testing PostGIS extension..."
    psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" -c "SELECT PostGIS_Version();"
    
    if [ $? -eq 0 ]; then
        echo "✅ PostGIS extension is available!"
    else
        echo "⚠️  PostGIS extension needs to be enabled"
        echo "Run: psql -c 'CREATE EXTENSION IF NOT EXISTS postgis;'"
    fi
else
    echo "❌ Database connection failed!"
    echo "Please check:"
    echo "1. Database endpoint in .env file"
    echo "2. Security group allows connections from bastion host"
    echo "3. Database is running and accessible"
fi
EOF

chmod +x /home/ubuntu/test-db-connection.sh

# Create application setup script
cat > /home/ubuntu/setup-application.sh << 'EOF'
#!/bin/bash
source /home/ubuntu/.env

echo "Setting up pgvector-hybrid-search-demo-py application..."

# Create application directory
mkdir -p /home/ubuntu/app
cd /home/ubuntu/app


# Python application setup
source /home/ubuntu/venv/bin/activate

# Create sample Flask application
cat > app.py << 'PYEOF'
from flask import Flask, jsonify
import psycopg2
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({
        "message": "Welcome to pgvector-hybrid-search-demo-py",
        "database": "Aurora PostgreSQL",
        "status": "running"
    })

@app.route('/db-test')
def test_db():
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return jsonify({
            "database_connected": True,
            "version": version
        })
    except Exception as e:
        return jsonify({
            "database_connected": False,
            "error": str(e)
        }), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
PYEOF

echo "Python application created. Run with: python app.py"


EOF

chmod +x /home/ubuntu/setup-application.sh

# Create database initialization script for Aurora PostgreSQL
cat > /home/ubuntu/init-database.sh << 'EOF'
#!/bin/bash
source /home/ubuntu/.env

echo "Initializing Aurora PostgreSQL database for pgvector-hybrid-search-demo-py..."

# Connect to database and run initialization
psql "postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME" << 'SQLEOF'
-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;


-- Create sample spatial tables for general use cases
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
('Sample Point 2', 'Second sample location', ST_GeomFromText('POINT(-74.0060 40.7128)', 4326)),
('Sample Point 3', 'Third sample location', ST_GeomFromText('POINT(-87.6298 41.8781)', 4326))
ON CONFLICT DO NOTHING;


-- Verify PostGIS installation
SELECT 'PostGIS Version: ' || PostGIS_Version() as info;
SELECT 'Total tables created: ' || count(*) as table_count FROM information_schema.tables WHERE table_schema = 'public';

echo 'Database initialization completed successfully!'
SQLEOF

if [ $? -eq 0 ]; then
    echo "✅ Database initialization completed successfully!"
else
    echo "❌ Database initialization failed!"
    exit 1
fi
EOF

chmod +x /home/ubuntu/init-database.sh

# Configure firewall for application access
sudo ufw allow 22    # SSH
sudo ufw allow 3000  # Application port
sudo ufw --force enable

# Set proper ownership
chown -R ubuntu:ubuntu /home/ubuntu/

echo ""
echo "🎉 Bastion host setup completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. Update database endpoint in /home/ubuntu/.env"
echo "2. Test database connection: ./test-db-connection.sh"
echo "3. Initialize database schema: ./init-database.sh"
echo "4. Set up application: ./setup-application.sh"
echo "5. Start application: cd app && python app.py"
echo ""
echo "🔗 Application will be available at: http://$(curl -s ifconfig.me):3000"
echo "📊 Database: Aurora PostgreSQL cluster"
echo "🛡️  Security: Bastion host configuration with Aurora PostgreSQL in private subnets"
