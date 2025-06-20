#!/bin/bash
# Comprehensive Setup Script for pgvector-hybrid-search-demo-py
# Target: Aurora PostgreSQL Aurora PostgreSQL 16
# Use Cases: Vector/Hybrid Search
# Language: Python

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== pgvector-hybrid-search-demo-py Setup Script ===${NC}"
echo -e "${BLUE}Target Platform: Aurora PostgreSQL${NC}"
echo -e "${BLUE}Use Cases: Vector/Hybrid Search${NC}"
echo ""

# Function to check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}❌ $1 is required but not installed.${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ $1 is available${NC}"
    fi
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"
check_command "psql"

check_command "python3"
check_command "pip3"


# Check AWS CLI for cloud deployment
if command -v aws &> /dev/null; then
    echo -e "${GREEN}✓ AWS CLI is available${NC}"
    AWS_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ AWS CLI not found (required for cloud deployment)${NC}"
    AWS_AVAILABLE=false
fi

# Environment setup
echo ""
echo -e "${YELLOW}Setting up environment...${NC}"

# Copy environment template if it doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env from template${NC}"
        echo -e "${YELLOW}⚠ Please update .env with your database credentials${NC}"
    else
        echo -e "${YELLOW}Creating .env file...${NC}"
        cat > .env << 'EOF'
# Database Configuration for pgvector-hybrid-search-demo-py
# Update these values with your actual database credentials

# For local development
DATABASE_URL=postgresql://postgres:password@localhost:5432/pgvector_hybrid_search_demo_py

# For AWS RDS/Aurora (uncomment and update when deploying)
# DB_HOST=your-database-endpoint.us-east-1.rds.amazonaws.com
# DB_PORT=5432
# DB_NAME=pgvector_hybrid_search_demo_py
# DB_USER=postgres
# DB_PASSWORD=your-secure-password

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=your-account-id

# Application Configuration
# Vector Search Configuration
OPENAI_API_KEY=your-openai-api-key
EMBEDDING_MODEL=text-embedding-ada-002
VECTOR_DIMENSIONS=384
EOF
        echo -e "${GREEN}✓ Created .env file with configuration template${NC}"
    fi
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Install dependencies
echo ""
echo -e "${YELLOW}Installing dependencies...${NC}"


# Python setup
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Created virtual environment${NC}"
fi

# Activate virtual environment
source venv/bin/activate
echo -e "${GREEN}✓ Activated virtual environment${NC}"

# Install Python packages
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip3 install --upgrade pip
    pip3 install -r requirements.txt
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ requirements.txt not found${NC}"
fi

# Verify database driver
python3 -c "import psycopg2; print('✓ PostgreSQL driver available')" 2>/dev/null || {
    echo -e "${RED}❌ PostgreSQL driver not available. Installing...${NC}"
    pip3 install psycopg2-binary
}


# Database connectivity check
echo ""
echo -e "${YELLOW}Checking database connectivity...${NC}"

# Source environment variables
if [ -f .env ]; then
    source .env
fi

# Test database connection
if [ -n "$DATABASE_URL" ]; then
    echo "Testing database connection..."
    if psql "$DATABASE_URL" -c "SELECT version();" &>/dev/null; then
        echo -e "${GREEN}✓ Database connection successful${NC}"
        DB_CONNECTED=true
    else
        echo -e "${YELLOW}⚠ Cannot connect to database. Please verify credentials in .env${NC}"
        DB_CONNECTED=false
    fi
else
    echo -e "${YELLOW}⚠ DATABASE_URL not configured in .env${NC}"
    DB_CONNECTED=false
fi

# Run database migrations if connected
if [ "$DB_CONNECTED" = true ]; then
    echo ""
    echo -e "${YELLOW}Running database migrations...${NC}"
    
    # Check if migration file exists and run it
    if [ -f "database/001_initial_schema.sql" ]; then
        echo "Applying initial schema migration..."
        psql "$DATABASE_URL" -f database/001_initial_schema.sql
        echo -e "${GREEN}✓ Database schema initialized${NC}"
    else
        echo -e "${YELLOW}⚠ Migration file not found. Database setup may be incomplete.${NC}"
    fi
    
    # Verify extensions
    echo "Verifying database extensions..."
    
    psql "$DATABASE_URL" -c "SELECT name, default_version, installed_version FROM pg_available_extensions WHERE name IN ('postgis', 'vector') AND installed_version IS NOT NULL;" 2>/dev/null || echo -e "${YELLOW}⚠ Some extensions may not be installed${NC}"
    
    
    # Show database info
    echo ""
    echo -e "${BLUE}Database Information:${NC}"
    psql "$DATABASE_URL" -c "
        SELECT 
            current_database() as database_name,
            current_user as connected_user,
            version() as postgresql_version;
    " 2>/dev/null || echo -e "${YELLOW}⚠ Could not retrieve database information${NC}"
fi

# Development server setup
echo ""
echo -e "${YELLOW}Setting up development server...${NC}"


# Python development setup
if [ -f "app.py" ] || [ -f "main.py" ] || [ -f "run.py" ]; then
    echo -e "${GREEN}✓ Python application files found${NC}"
    PYTHON_APP=true
else
    echo -e "${YELLOW}⚠ Python application entry point not found${NC}"
    PYTHON_APP=false
fi


# Create helpful scripts
echo ""
echo -e "${YELLOW}Creating helper scripts...${NC}"

# Create run script
cat > run_demo.sh << 'EOF'
#!/bin/bash
# Quick start script for pgvector-hybrid-search-demo-py

set -e
source .env 2>/dev/null || echo "Warning: .env file not found"


# Activate Python virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the main application
if [ -f "app.py" ]; then
    python3 app.py
elif [ -f "main.py" ]; then
    python3 main.py
elif [ -f "run.py" ]; then
    python3 run.py
else
    echo "No main Python file found. Try running modules individually:"
    echo "python3 module_01.py"
fi

EOF

chmod +x run_demo.sh
echo -e "${GREEN}✓ Created run_demo.sh script${NC}"

# Create database reset script
cat > reset_database.sh << 'EOF'
#!/bin/bash
# Database reset script for pgvector-hybrid-search-demo-py

set -e
source .env 2>/dev/null || { echo "Error: .env file not found"; exit 1; }

if [ -z "$DATABASE_URL" ]; then
    echo "Error: DATABASE_URL not set in .env"
    exit 1
fi

echo "⚠ This will drop and recreate all database objects!"
read -p "Are you sure? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Resetting database..."
    
    # Drop schemas
    psql "$DATABASE_URL" -c "DROP SCHEMA IF EXISTS app_data CASCADE;"
    psql "$DATABASE_URL" -c "DROP SCHEMA IF EXISTS analytics CASCADE;"
    psql "$DATABASE_URL" -c "DROP SCHEMA IF EXISTS logs CASCADE;"
    
    # Re-run migrations
    if [ -f "database/001_initial_schema.sql" ]; then
        psql "$DATABASE_URL" -f database/001_initial_schema.sql
        echo "✓ Database reset complete"
    else
        echo "❌ Migration file not found"
    fi
else
    echo "Database reset cancelled"
fi
EOF

chmod +x reset_database.sh
echo -e "${GREEN}✓ Created reset_database.sh script${NC}"

# Final summary
echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Update .env with your database credentials"
echo "2. Ensure your Aurora PostgreSQL instance is running"
3. Set your OpenAI API key in .env (for vector embeddings)
echo "3. Run the demo: ./run_demo.sh"
echo "4. Or run modules individually: python3 module_01.py"
echo ""
echo -e "${BLUE}Available Scripts:${NC}"
echo "• ./run_demo.sh - Start the main application"
echo "• ./reset_database.sh - Reset database to initial state"
echo ""
echo -e "${BLUE}Use Cases Configured:${NC}"
echo "• Vector/Hybrid Search"
echo ""

echo -e "${YELLOW}pgvector Setup:${NC} Vector similarity search enabled"



echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
