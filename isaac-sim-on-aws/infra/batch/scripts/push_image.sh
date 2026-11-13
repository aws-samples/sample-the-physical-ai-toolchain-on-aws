#!/bin/bash
# Build and push Isaac Sim container to ECR.
set -e

REGION=${AWS_REGION:-us-east-1}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/isaac-sim-batch"

echo "=== Building Isaac Sim Container ==="

# Login to NGC (password via stdin to avoid exposure in process list)
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin

# Login to ECR
aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin \
    ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

# Build and push (target linux/amd64 for EC2)
docker build --platform linux/amd64 -t isaac-sim-batch:v1 -f container/Dockerfile .
docker tag isaac-sim-batch:v1 ${REPO}:v1
docker push ${REPO}:v1

echo "=== Pushed: ${REPO}:v1 ==="
