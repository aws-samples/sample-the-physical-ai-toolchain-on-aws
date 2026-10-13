#!/usr/bin/env bash
# Launch a Cosmos Transfer 2.5 NIM server on EC2 p5.48xlarge via Capacity Block.
#
# Automates all of Lab 4's infrastructure setup:
#   1. Scans for available Capacity Blocks across regions
#   2. Purchases a block (with confirmation)
#   3. Waits for activation
#   4. Launches p5.48xlarge into the block
#   5. Pulls the NIM container from NGC
#   6. Starts the Transfer NIM server
#   7. Waits for readiness
#
# Prerequisites:
#   - AWS CLI configured with EC2/IAM/SSM/SecretsManager permissions
#   - NGC API key in Secrets Manager (physical-ai/ngc-api-key)
#   - HuggingFace token in Secrets Manager (physical-ai/hf-token)
#   - HF licenses accepted: nvidia/Cosmos-Guardrail1, nvidia/Cosmos-1.0-Guardrail,
#     nvidia/Cosmos-Predict2.5-2B, nvidia/Cosmos-Transfer2.5-2B
#
# Usage:
#   bash scripts/cosmos-transfer-launch.sh              # full deploy
#   bash scripts/cosmos-transfer-launch.sh --dry-run    # show plan, no AWS calls
#   bash scripts/cosmos-transfer-launch.sh --status     # check existing server
#
# Output: prints INSTANCE_ID when ready. Use for inference:
#   export COSMOS_TRANSFER_INSTANCE=<instance-id>
#   # Then run the Python inference script (see Lab 4 Step 5)

set -euo pipefail

REGIONS="${COSMOS_REGIONS:-us-east-1 us-east-2 us-west-2}"
INSTANCE_TYPE="p5.48xlarge"
ROLE_NAME="physical-ai-dev-cosmos-role"
PROFILE_NAME="physical-ai-dev-cosmos-profile"
SG_NAME="cosmos-test-sg"
NGC_SECRET="physical-ai/ngc-api-key"
HF_SECRET="physical-ai/hf-token"
# Secrets live in the Foundation region, which can differ from the cheapest Capacity Block region.
SECRETS_REGION="${SECRETS_REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region || true)}}}"
if [ -z "$SECRETS_REGION" ]; then
  echo "ERROR: set SECRETS_REGION (or AWS_REGION) to the Foundation stack region." >&2
  exit 1
fi

DRY_RUN=false
STATUS_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --status) STATUS_ONLY=true ;;
  esac
done

echo "============================================================"
echo "  Cosmos Transfer 2.5 NIM Server"
echo "  Instance: $INSTANCE_TYPE (H100 80GB required)"
echo "============================================================"

if [ "$DRY_RUN" = true ]; then
  echo ""
  echo "[DRY RUN] Would:"
  echo "  1. Scan Capacity Blocks in: $REGIONS"
  echo "  2. Purchase the cheapest/soonest block"
  echo "  3. Launch $INSTANCE_TYPE into the block"
  echo "  4. Pull nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest (~56 GB)"
  echo "  5. Start the NIM server (auto-selects H100 FP8 latency profile)"
  echo "  6. Wait for /v1/health/ready"
  echo ""
  echo "  Estimated cost: ~\$37/hr (Capacity Block pricing)"
  echo "  Estimated time to ready: ~35 min"
  echo "  Prerequisites: NGC key + HF token in Secrets Manager"
  exit 0
fi

# --- Step 1: Verify IAM (managed by Terraform in foundation/infra/) ---
echo ""
echo "Step 1: Verifying IAM role + instance profile..."
if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" > /dev/null 2>&1; then
  echo "  ERROR: Instance profile '$PROFILE_NAME' not found."
  echo "  Deploy foundation first: cd foundation/infra && terraform apply"
  exit 1
fi
echo "  ✓ IAM ready (managed by Terraform)"

# --- Step 2: Scan for Capacity Blocks ---
echo ""
echo "Step 2: Scanning for Capacity Blocks..."
BEST_REGION=""
BEST_OFFERING=""
BEST_PRICE=999999

for REGION in $REGIONS; do
  OFFERINGS=$(aws ec2 describe-capacity-block-offerings \
    --instance-type "$INSTANCE_TYPE" --capacity-duration-hours 24 --instance-count 1 \
    --region "$REGION" --output json 2>/dev/null || echo '{"CapacityBlockOfferings":[]}')

  OFFERING=$(echo "$OFFERINGS" | python3 -c "
import json, sys
data = json.load(sys.stdin)
offers = data.get('CapacityBlockOfferings', [])
if offers:
    best = min(offers, key=lambda x: float(x.get('UpfrontFee', '999999')))
    print(json.dumps(best))
" 2>/dev/null || true)

  if [ -n "$OFFERING" ]; then
    PRICE=$(echo "$OFFERING" | python3 -c "import json,sys; print(json.load(sys.stdin).get('UpfrontFee','999999'))")
    echo "  $REGION: \$$PRICE"
    if [ "$(echo "$PRICE < $BEST_PRICE" | bc 2>/dev/null || echo 0)" = "1" ] || [ "$BEST_REGION" = "" ]; then
      BEST_REGION="$REGION"
      BEST_OFFERING="$OFFERING"
      BEST_PRICE="$PRICE"
    fi
  else
    echo "  $REGION: none available"
  fi
done

if [ -z "$BEST_OFFERING" ]; then
  echo ""
  echo "  ERROR: No Capacity Blocks available in any region. Try again later."
  exit 1
fi

OFFERING_ID=$(echo "$BEST_OFFERING" | python3 -c "import json,sys; print(json.load(sys.stdin)['CapacityBlockOfferingId'])")
BLOCK_AZ=$(echo "$BEST_OFFERING" | python3 -c "import json,sys; print(json.load(sys.stdin)['AvailabilityZone'])")
START=$(echo "$BEST_OFFERING" | python3 -c "import json,sys; print(json.load(sys.stdin)['StartDate'])")
DURATION=$(echo "$BEST_OFFERING" | python3 -c "import json,sys; print(json.load(sys.stdin)['CapacityBlockDurationHours'])")

echo ""
echo "  Best: $OFFERING_ID in $BLOCK_AZ ($BEST_REGION)"
echo "  Duration: ${DURATION}h, Cost: \$$BEST_PRICE, Starts: $START"
read -p "  Purchase this block? (y/n): " CONFIRM
[ "$CONFIRM" != "y" ] && exit 0

# --- Step 3: Purchase ---
CR_ID=$(aws ec2 purchase-capacity-block --capacity-block-offering-id "$OFFERING_ID" \
  --instance-platform Linux/UNIX --region "$BEST_REGION" \
  --tag-specifications 'ResourceType=capacity-reservation,Tags=[{Key=Name,Value=cosmos-transfer},{Key=Project,Value=physical-ai}]' \
  --query 'CapacityReservation.CapacityReservationId' --output text)
echo "  ✓ Purchased: $CR_ID"

# --- Step 4: Wait for activation ---
echo ""
echo "Step 3: Waiting for block to activate..."
while true; do
  STATE=$(aws ec2 describe-capacity-reservations --capacity-reservation-ids "$CR_ID" \
    --region "$BEST_REGION" --query 'CapacityReservations[0].State' --output text)
  echo "  State: $STATE ($(date +%H:%M:%S))"
  [ "$STATE" = "active" ] && break
  sleep 30
done
echo "  ✓ Block is ACTIVE"

# --- Step 5: Launch ---
echo ""
echo "Step 4: Launching $INSTANCE_TYPE..."
SUBNET=$(aws ec2 describe-subnets --filters "Name=availability-zone,Values=$BLOCK_AZ" \
  --query 'Subnets[0].SubnetId' --output text --region "$BEST_REGION")
AMI=$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --region "$BEST_REGION" --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text)

VPC=$(aws ec2 describe-subnets --subnet-ids "$SUBNET" --region "$BEST_REGION" --query 'Subnets[0].VpcId' --output text)
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC" \
  --region "$BEST_REGION" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" --description "Cosmos Transfer" \
    --vpc-id "$VPC" --region "$BEST_REGION" --query 'GroupId' --output text)
fi

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
  --placement AvailabilityZone="$BLOCK_AZ" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name="$PROFILE_NAME" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3"}}]' \
  --instance-market-options '{"MarketType":"capacity-block"}' \
  --capacity-reservation-specification '{"CapacityReservationTarget":{"CapacityReservationId":"'$CR_ID'"}}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cosmos-transfer},{Key=Project,Value=physical-ai}]' \
  --region "$BEST_REGION" --query 'Instances[0].InstanceId' --output text)

echo "  Instance: $INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$BEST_REGION"
echo "  ✓ Running"

# --- Step 6: Setup NIM ---
echo ""
echo "Step 5: Setting up Transfer NIM (via SSM)..."
echo "  Waiting for SSM..."
while true; do
  PING=$(aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region "$BEST_REGION" --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || echo "None")
  [ "$PING" = "Online" ] && break
  sleep 10
done
echo "  ✓ SSM online"

echo "  NGC login + NIM pull (~20 min)..."
aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
  --parameters '{"commands":["aws secretsmanager get-secret-value --secret-id physical-ai/ngc-api-key --region '"$SECRETS_REGION"' --query SecretString --output text | docker login nvcr.io --username \\$oauthtoken --password-stdin && docker pull nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest"]}' \
  --timeout-seconds 1800 --region "$BEST_REGION" > /dev/null

echo "  Waiting for pull to complete..."
sleep 1200  # 20 min for 56 GB pull

echo "  Starting NIM server..."
aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
  --parameters '{"commands":["NGC_KEY=$(aws secretsmanager get-secret-value --secret-id physical-ai/ngc-api-key --region '"$SECRETS_REGION"' --query SecretString --output text) && HF_TOKEN=$(aws secretsmanager get-secret-value --secret-id physical-ai/hf-token --region '"$SECRETS_REGION"' --query SecretString --output text) && docker run -d --name cosmos-transfer-nim --gpus all --ipc=host --shm-size=64g -p 8000:8000 -e NGC_API_KEY=$NGC_KEY -e HF_TOKEN=$HF_TOKEN nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:latest"]}' \
  --region "$BEST_REGION" > /dev/null

# --- Step 7: Wait for ready ---
echo ""
echo "Step 6: Waiting for model load (~15 min)..."
while true; do
  HEALTH=$(aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
    --parameters '{"commands":["curl -s http://localhost:8000/v1/health/ready 2>/dev/null && echo READY || echo NOT_READY"]}' \
    --region "$BEST_REGION" --query 'Command.CommandId' --output text)
  sleep 15
  RESULT=$(aws ssm get-command-invocation --command-id "$HEALTH" --instance-id "$INSTANCE_ID" \
    --region "$BEST_REGION" --query 'StandardOutputContent' --output text 2>/dev/null || echo "")
  if echo "$RESULT" | grep -q "READY"; then
    break
  fi
  echo "  Still loading... ($(date +%H:%M:%S))"
  sleep 60
done

echo ""
echo "============================================================"
echo "  ✅ COSMOS TRANSFER NIM IS READY"
echo ""
echo "  Instance: $INSTANCE_ID"
echo "  Region:   $BEST_REGION"
echo "  Endpoint: http://localhost:8000/v1/infer (via SSM port-forward)"
echo ""
echo "  To run inference, see Lab 4 Step 5 or:"
echo "    export COSMOS_TRANSFER_INSTANCE=$INSTANCE_ID"
echo "    export COSMOS_TRANSFER_REGION=$BEST_REGION"
echo ""
echo "  To terminate when done:"
echo "    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $BEST_REGION"
echo "============================================================"
