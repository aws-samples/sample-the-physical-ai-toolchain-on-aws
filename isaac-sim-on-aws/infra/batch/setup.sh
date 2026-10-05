#!/bin/bash
# Automated environment setup for Isaac Sim AWS Batch pipeline (Terraform).
set -e

echo "=== Isaac Sim AWS Batch — Setup ==="

# Check prerequisites
command -v python3   >/dev/null 2>&1 || { echo "Python 3.12+ required"; exit 1; }
command -v docker    >/dev/null 2>&1 || { echo "Docker required"; exit 1; }
command -v aws       >/dev/null 2>&1 || { echo "AWS CLI required"; exit 1; }
command -v terraform >/dev/null 2>&1 || { echo "Terraform >= 1.5 required (https://developer.hashicorp.com/terraform/install)"; exit 1; }

# Python environment (submit scripts, simulation helpers, config generation)
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install Python dependencies
pip install -q -r requirements.txt

# Verify OpenUSD
python3 -c "from pxr import Usd; print('OpenUSD OK')" 2>/dev/null || \
    echo "WARNING: OpenUSD not available locally. Scene composition requires usd-core."

# Set environment
export AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "NOT_SET")
export AWS_REGION=${AWS_REGION:-us-east-1}
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-$AWS_REGION}

echo ""
echo "=== Setup Complete ==="
echo "Account: ${AWS_ACCOUNT}"
echo "Region:  ${AWS_REGION}"
echo ""
echo "Next steps (Terraform root is this directory):"
echo "  source .venv/bin/activate"
echo "  terraform init"
echo "  terraform apply"
