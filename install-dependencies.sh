#!/bin/bash
# Comprehensive Dependency Installation Script for pgvector-hybrid-search-demo-py
# This script installs all required system and Python dependencies

set -e

echo "🚀 Installing dependencies for pgvector-hybrid-search-demo-py..."

# Function to detect OS and install system dependencies
install_system_dependencies() {
    echo "📦 Installing system dependencies..."
    
    if command -v apt-get >/dev/null 2>&1; then
        # Ubuntu/Debian
        echo "Detected Ubuntu/Debian system"
        export DEBIAN_FRONTEND=noninteractive
        sudo apt-get update -qq
        sudo apt-get install -y \
            python3 \
            python3-pip \
            python3-venv \
            python3-dev \
            postgresql-client \
            libpq-dev \
            build-essential \
            curl \
            unzip \
            git \
            wget
        
        # Install AWS CLI v2 for Ubuntu/Debian
        if ! command -v aws >/dev/null 2>&1; then
            echo "Installing AWS CLI v2..."
            curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
            unzip -q awscliv2.zip
            sudo ./aws/install
            rm -rf awscliv2.zip aws/
        fi
        
    elif command -v yum >/dev/null 2>&1; then
        # Amazon Linux/RHEL/CentOS
        echo "Detected Amazon Linux/RHEL/CentOS system"
        sudo yum update -y -q
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y \
            python3 \
            python3-pip \
            python3-devel \
            postgresql \
            postgresql-devel \
            curl \
            unzip \
            git \
            wget
        
        # Install AWS CLI v2 for Amazon Linux
        if ! command -v aws >/dev/null 2>&1; then
            echo "Installing AWS CLI v2..."
            curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
            unzip -q awscliv2.zip
            sudo ./aws/install
            rm -rf awscliv2.zip aws/
        fi
        
    elif command -v brew >/dev/null 2>&1; then
        # macOS with Homebrew
        echo "Detected macOS system"
        brew update
        brew install \
            python3 \
            postgresql \
            libpq \
            awscli
        
    else
        echo "⚠️  Unknown package manager. Please install manually:"
        echo "   - Python 3.8+ with pip and development headers"
        echo "   - PostgreSQL client tools (psql)"
        echo "   - PostgreSQL development libraries (libpq-dev)"
        echo "   - Build tools (gcc, make)"
        echo "   - AWS CLI v2"
        echo "   - curl, unzip, git, wget"
        exit 1
    fi
}

# Function to verify system dependencies
verify_system_dependencies() {
    echo "🔍 Verifying system dependencies..."
    local missing_deps=0
    
    if ! command -v python3 >/dev/null 2>&1; then
        echo "❌ Python 3 not found"
        missing_deps=1
    else
        echo "✅ Python 3 found: $(python3 --version)"
    fi
    
    if ! command -v psql >/dev/null 2>&1; then
        echo "❌ PostgreSQL client not found"
        missing_deps=1
    else
        echo "✅ PostgreSQL client found: $(psql --version)"
    fi
    
    if ! command -v pip3 >/dev/null 2>&1 && ! command -v pip >/dev/null 2>&1; then
        echo "❌ pip not found"
        missing_deps=1
    else
        echo "✅ pip found"
    fi
    
    if ! command -v aws >/dev/null 2>&1; then
        echo "❌ AWS CLI not found"
        missing_deps=1
    else
        echo "✅ AWS CLI found: $(aws --version)"
    fi
    
    return $missing_deps
}

# Main installation process
main() {
    echo "Starting dependency installation for pgvector-hybrid-search-demo-py..."
    echo "======================================================="
    
    # Check if system dependencies are already installed
    if ! verify_system_dependencies; then
        echo "Installing missing system dependencies..."
        install_system_dependencies
        
        # Verify again after installation
        if ! verify_system_dependencies; then
            echo "❌ Failed to install some system dependencies"
            exit 1
        fi
    fi
    
    echo "✅ All system dependencies verified"
    
    # Create Python virtual environment
    echo "🐍 Setting up Python virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "✅ Virtual environment created"
    else
        echo "✅ Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip and install build tools
    echo "📦 Upgrading pip and installing build tools..."
    python -m pip install --upgrade pip setuptools wheel
    
    # Install Python dependencies
    echo "📦 Installing Python dependencies from requirements.txt..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        echo "✅ Python dependencies installed successfully"
    else
        echo "⚠️  requirements.txt not found, installing core dependencies..."
        pip install flask psycopg2-binary python-dotenv requests
    fi
    
    # Test core imports
    echo "🧪 Testing Python imports..."
    python -c "
import sys
try:
    import flask
    print('✅ Flask imported successfully')
except ImportError as e:
    print(f'❌ Flask import failed: {e}')
    sys.exit(1)

try:
    import psycopg2
    print('✅ psycopg2 imported successfully')
except ImportError as e:
    print(f'❌ psycopg2 import failed: {e}')
    sys.exit(1)

try:
    import json
    import os
    print('✅ Standard library imports successful')
except ImportError as e:
    print(f'❌ Standard library import failed: {e}')
    sys.exit(1)

print('✅ All core dependencies imported successfully')
"
    
    if [ $? -ne 0 ]; then
        echo "❌ Python dependency test failed"
        exit 1
    fi
    
    # Create environment file if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            echo "📝 Created .env file from template"
            echo "⚠️  Please update .env with your actual configuration"
        else
            echo "⚠️  No .env.example found, please create .env manually"
        fi
    fi
    
    # Make scripts executable
    echo "🔧 Making scripts executable..."
    chmod +x *.sh 2>/dev/null || true
    chmod +x scripts/*.sh 2>/dev/null || true
    
    echo ""
    echo "🎉 Dependency installation completed successfully!"
    echo "======================================================="
    echo ""
    echo "📋 Next Steps:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Update .env with your database connection details"
    echo "3. Run setup script: ./setup.sh"
    echo "4. Start application: python app.py"
    echo ""
    echo "🔗 Application will be available at: http://localhost:5000"
    echo "📊 Database: PostgreSQL with PostGIS"
    echo "☁️  Cloud: AWS infrastructure via CloudFormation"
}

# Run main function
main "$@"
