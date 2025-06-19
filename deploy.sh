#!/bin/bash
# =============================================================================
# AWS CloudFormation Deployment Script for pgvector-hybrid-search-demo-py
# =============================================================================
# Description: Automated deployment of complete PostgreSQL demo environment
# Target: Aurora PostgreSQL 16 with PostGIS
# Use Case: Vector/Hybrid Search
# Language: Python
# =============================================================================

set -e  # Exit on any error
set -u  # Exit on undefined variables
set -o pipefail  # Exit on pipe failures

# Color codes for output
RED='\x1b[0;31m'
GREEN='\x1b[0;32m'
YELLOW='\x1b[1;33m'
BLUE='\x1b[0;34m'
NC='\x1b[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Error handling
error_exit() {
    log_error "$1"
    log_error "Deployment failed. Check the logs above for details."
    exit 1
}

# Configuration variables
PROJECT_NAME="pgvector-hybrid-search-demo-py"
STACK_NAME="${PROJECT_NAME}-stack"
AWS_REGION="${AWS_REGION:-us-west-2}"
KEY_PAIR_NAME="${KEY_PAIR_NAME:-${PROJECT_NAME}-keypair}"
POSTGRES_VERSION="16"
DB_INSTANCE_TYPE="db.r6g.large"
BASTION_INSTANCE_TYPE="t3.micro"

log_info "Starting AWS deployment for pgvector-hybrid-search-demo-py"
log_info "Configuration:"
log_info "  - Project: $PROJECT_NAME"
log_info "  - Stack: $STACK_NAME" 
log_info "  - Region: $AWS_REGION"
log_info "  - Database: Aurora PostgreSQL $POSTGRES_VERSION"
log_info "  - Instance Type: $DB_INSTANCE_TYPE"

# Prerequisites check
log_info "Checking prerequisites..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    error_exit "AWS CLI is required but not installed. Please install AWS CLI v2."
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    error_exit "AWS credentials not configured. Run 'aws configure' first."
fi

# Verify region is valid
if ! aws ec2 describe-regions --region-names $AWS_REGION &> /dev/null; then
    error_exit "Invalid AWS region: $AWS_REGION"
fi

log_success "Prerequisites check passed"

# Create EC2 Key Pair if it doesn't exist
log_info "Checking EC2 Key Pair..."
if ! aws ec2 describe-key-pairs --key-names $KEY_PAIR_NAME --region $AWS_REGION &> /dev/null; then
    log_info "Creating new EC2 Key Pair: $KEY_PAIR_NAME"
    aws ec2 create-key-pair \
        --key-name $KEY_PAIR_NAME \
        --query 'KeyMaterial' \
        --output text \
        --region $AWS_REGION > ${KEY_PAIR_NAME}.pem
    
    chmod 400 ${KEY_PAIR_NAME}.pem
    log_success "Key pair created: ${KEY_PAIR_NAME}.pem"
else
    log_info "Key pair already exists: $KEY_PAIR_NAME"
fi

# Validate CloudFormation template
log_info "Validating CloudFormation template..."
if ! aws cloudformation validate-template \
    --template-body file://cloudformation/main.yaml \
    --region $AWS_REGION &> /dev/null; then
    error_exit "CloudFormation template validation failed"
fi
log_success "CloudFormation template is valid"

# Check for existing stack
if aws cloudformation describe-stacks --stack-name $STACK_NAME --region $AWS_REGION &> /dev/null; then
    log_warning "Stack $STACK_NAME already exists"
    read -p "Do you want to update the existing stack? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        OPERATION="update"
    else
        log_info "Deployment cancelled by user"
        exit 0
    fi
else
    OPERATION="create"
fi

# Deploy CloudFormation stack
log_info "Deploying CloudFormation stack..."
if [ "$OPERATION" = "create" ]; then
    aws cloudformation create-stack \
        --stack-name $STACK_NAME \
        --template-body file://cloudformation/main.yaml \
        --parameters \
            ParameterKey=ProjectName,ParameterValue=$PROJECT_NAME \
            ParameterKey=KeyPairName,ParameterValue=$KEY_PAIR_NAME \
            ParameterKey=PostgreSQLVersion,ParameterValue=$POSTGRES_VERSION \
            ParameterKey=DatabaseInstanceType,ParameterValue=$DB_INSTANCE_TYPE \
            ParameterKey=BastionInstanceType,ParameterValue=$BASTION_INSTANCE_TYPE \
        --capabilities CAPABILITY_IAM \
        --region $AWS_REGION \
        --tags \
            Key=Project,Value=$PROJECT_NAME \
            Key=Environment,Value=demo \
            Key=UseCase,Value="Vector/Hybrid Search" \
        || error_exit "Stack creation failed"
else
    aws cloudformation update-stack \
        --stack-name $STACK_NAME \
        --template-body file://cloudformation/main.yaml \
        --parameters \
            ParameterKey=ProjectName,ParameterValue=$PROJECT_NAME \
            ParameterKey=KeyPairName,ParameterValue=$KEY_PAIR_NAME \
            ParameterKey=PostgreSQLVersion,ParameterValue=$POSTGRES_VERSION \
            ParameterKey=DatabaseInstanceType,ParameterValue=$DB_INSTANCE_TYPE \
            ParameterKey=BastionInstanceType,ParameterValue=$BASTION_INSTANCE_TYPE \
        --capabilities CAPABILITY_IAM \
        --region $AWS_REGION \
        || error_exit "Stack update failed"
fi

# Wait for stack completion
log_info "Waiting for stack $OPERATION to complete (this may take 10-15 minutes)..."
if [ "$OPERATION" = "create" ]; then
    aws cloudformation wait stack-create-complete \
        --stack-name $STACK_NAME \
        --region $AWS_REGION
else
    aws cloudformation wait stack-update-complete \
        --stack-name $STACK_NAME \
        --region $AWS_REGION
fi

# Check stack status
STACK_STATUS=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query 'Stacks[0].StackStatus' \
    --output text \
    --region $AWS_REGION)

if [[ "$STACK_STATUS" == *"COMPLETE" ]]; then
    log_success "Stack $OPERATION completed successfully"
else
    error_exit "Stack $OPERATION failed with status: $STACK_STATUS"
fi

# Get stack outputs
log_info "Retrieving stack outputs..."
BASTION_IP=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query 'Stacks[0].Outputs[?OutputKey==`BastionHostIP`].OutputValue' \
    --output text \
    --region $AWS_REGION)

DB_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query 'Stacks[0].Outputs[?OutputKey==`DatabaseEndpoint`].OutputValue' \
    --output text \
    --region $AWS_REGION)

VPC_ID=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query 'Stacks[0].Outputs[?OutputKey==`VpcId`].OutputValue' \
    --output text \
    --region $AWS_REGION)

# Validate outputs
if [ -z "$BASTION_IP" ] || [ -z "$DB_ENDPOINT" ] || [ -z "$VPC_ID" ]; then
    error_exit "Failed to retrieve required stack outputs"
fi

# Save connection information
cat > connection-info.txt << EOF
# pgvector-hybrid-search-demo-py AWS Deployment Information
# Generated: $(date)

## Stack Details
Stack Name: $STACK_NAME
AWS Region: $AWS_REGION
VPC ID: $VPC_ID

## Bastion Host
IP Address: $BASTION_IP
SSH Command: ssh -i ${KEY_PAIR_NAME}.pem ubuntu@$BASTION_IP
Key File: ${KEY_PAIR_NAME}.pem

## Database
Type: Aurora PostgreSQL $POSTGRES_VERSION
Endpoint: $DB_ENDPOINT
Port: 5432
Database: postgres
Username: postgres

## Next Steps
1. Connect to bastion host: ssh -i ${KEY_PAIR_NAME}.pem ubuntu@$BASTION_IP
2. Update database endpoint in /home/ubuntu/.env
3. Run database initialization: ./init-database.sh
4. Set up application: ./setup-application.sh
5. Start application and access at: http://$BASTION_IP:5000

## Cleanup
To delete all resources: aws cloudformation delete-stack --stack-name $STACK_NAME --region $AWS_REGION
EOF

# Test bastion host connectivity
log_info "Testing bastion host connectivity..."
if timeout 10 nc -z $BASTION_IP 22; then
    log_success "Bastion host is accessible on port 22"
else
    log_warning "Bastion host is not yet accessible (may take a few more minutes)"
fi

# Create environment file template for bastion host
cat > bastion-env-template.txt << EOF
# Environment file template for ${PROJECT_NAME}
# Copy this to /home/ubuntu/.env on the bastion host

DB_HOST=$DB_ENDPOINT
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=SecurePassword123!
DATABASE_URL=postgresql://postgres:SecurePassword123!@$DB_ENDPOINT:5432/postgres
AWS_REGION=$AWS_REGION
PROJECT_NAME=$PROJECT_NAME
EOF

# Create quick setup script for bastion host
cat > quick-setup.sh << 'EOF'
#!/bin/bash
# Quick setup script to run on bastion host

# Copy environment file
cp bastion-env-template.txt /home/ubuntu/.env

# Run bastion setup
chmod +x /home/ubuntu/scripts/bastion-setup.sh
/home/ubuntu/scripts/bastion-setup.sh

# Initialize database
chmod +x /home/ubuntu/init-database.sh
/home/ubuntu/init-database.sh

# Setup application
chmod +x /home/ubuntu/setup-application.sh
/home/ubuntu/setup-application.sh

echo "Setup completed! Application available at: http://$(curl -s ifconfig.me):5000"
EOF

chmod +x quick-setup.sh

# Summary
log_success "AWS deployment completed successfully!"
echo ""
echo "==================================================================="
echo "                     DEPLOYMENT SUMMARY"
echo "==================================================================="
echo "Project: $PROJECT_NAME"
echo "Stack: $STACK_NAME"
echo "Region: $AWS_REGION"
echo "Database: Aurora PostgreSQL $POSTGRES_VERSION"
echo ""
echo "Bastion Host IP: $BASTION_IP"
echo "Database Endpoint: $DB_ENDPOINT"
echo ""
echo "SSH Connection:"
echo "  ssh -i ${KEY_PAIR_NAME}.pem ubuntu@$BASTION_IP"
echo ""
echo "Files created:"
echo "  - ${KEY_PAIR_NAME}.pem (SSH key)"
echo "  - connection-info.txt (detailed connection info)"
echo "  - bastion-env-template.txt (environment variables)"
echo "  - quick-setup.sh (automated setup script)"
echo ""
echo "Next steps:"
echo "  1. Upload files to bastion host"
echo "  2. Run initialization scripts"
echo "  3. Access application at: http://$BASTION_IP:5000"
echo ""
echo "==================================================================="

# Optional: Upload setup files to bastion host
read -p "Do you want to upload setup files to the bastion host now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Uploading setup files to bastion host..."
    
    # Wait for bastion host to be fully ready
    log_info "Waiting for bastion host to be fully ready..."
    sleep 60
    
    # Upload files
    scp -o StrictHostKeyChecking=no -i ${KEY_PAIR_NAME}.pem \
        bastion-env-template.txt \
        quick-setup.sh \
        ubuntu@$BASTION_IP:~/
    
    scp -o StrictHostKeyChecking=no -i ${KEY_PAIR_NAME}.pem -r \
        scripts/ \
        cloudformation/ \
        ubuntu@$BASTION_IP:~/
    
    log_success "Files uploaded successfully"
    log_info "Connect to bastion host and run: chmod +x quick-setup.sh && ./quick-setup.sh"
fi

log_success "Deployment script completed successfully!"
