#!/usr/bin/env bash
# SPDX-License-Identifier: MIT-0
#
# Per-account build of the Isaac GR00T N1.7 training container.
#
# Builds containers/gr00t-training/Dockerfile.n17 and pushes it to THIS account's ECR
# repository, so every workshop participant account ends up with its own copy of the
# ~22.6 GB image and no attendee ever has to build it (or hold an NGC credential)
# during the session. Designed to run unattended in CodeBuild (see the workshop
# CloudFormation template, which clones this repo and invokes this script), but it
# works from any shell with credentials and a privileged Docker daemon.
#
# Two properties matter for a 100-account scale event:
#   * Idempotent - safe to re-run; the tags are simply overwritten (:n17) or rewritten
#                  identically (:n17-YYYYMMDD within the same UTC day).
#   * Observable  - fails loudly with a non-zero exit so a per-account health check
#                   (`aws codebuild batch-get-builds`) tells you which accounts are ready.
#
# Environment:
#   ECR_REPO_URI           (required) Destination ECR repository URI, WITHOUT a tag, e.g.
#                          <account>.dkr.ecr.<region>.amazonaws.com/physical-ai/gr00t-training
#                          This script appends :n17 and :n17-YYYYMMDD itself.
#   AWS_DEFAULT_REGION     (required) Region used for every AWS CLI call and both ECR logins.
#   NGC_SECRET_ID          Secrets Manager secret id/ARN holding the NGC API key, used to log
#                          in to nvcr.io before pulling the NGC base image. Optional: the
#                          nvcr.io/nvidia/pytorch base also pulls anonymously. When it IS set
#                          the key must be readable and accepted, or the build fails loudly.
#   CENTRAL_BASE_IMAGE_URI Pre-mirrored base image in a central ECR. Optional, and takes
#                          PRECEDENCE over NGC_SECRET_ID: when it is set no NGC login happens
#                          at all, so participant accounts never need an NGC key.
#
# With neither set, the NGC base image is pulled from nvcr.io anonymously. For a
# many-account event prefer CENTRAL_BASE_IMAGE_URI, so that every account does not pull
# the base from nvcr.io itself.

set -euo pipefail

: "${ECR_REPO_URI:?set ECR_REPO_URI}"
: "${AWS_DEFAULT_REGION:?set AWS_DEFAULT_REGION}"
NGC_SECRET_ID="${NGC_SECRET_ID:-}"
CENTRAL_BASE_IMAGE_URI="${CENTRAL_BASE_IMAGE_URI:-}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# NOTE: do not pipe any long-running command in this script into `head`. Under
# `set -o pipefail`, head closing the pipe early sends SIGPIPE to the producer and the
# script dies with exit 141 - that exact bug bit cosmos-on-aws/event-provision.sh as soon
# as terraform printed one extra line. Narrow output with --format, --query or tail.

#-----------------------------------------------------------------------------------
# 1. Locate the repo root.
#
# CodeBuild invokes this script by absolute path with an unrelated working directory, so
# the context has to be derived rather than assumed. The docker build context MUST be
# $REPO_ROOT because Dockerfile.n17 does
#   COPY containers/gr00t-training/train_entrypoint_n17.py ...
#   COPY containers/gr00t-training/batch-train-entrypoint-n17.sh ...
# - repo-root-relative paths that break under any other context.
#-----------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
log "Repo root: ${REPO_ROOT}"
log "Destination repository: ${ECR_REPO_URI}"

#-----------------------------------------------------------------------------------
# 2. Log in to this account's ECR (the push target).
#-----------------------------------------------------------------------------------
REGISTRY="$(echo "$ECR_REPO_URI" | cut -d/ -f1)"
log "Logging in to ECR registry ${REGISTRY}"
aws ecr get-login-password --region "$AWS_DEFAULT_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

#-----------------------------------------------------------------------------------
# 3. Select exactly one base-image source.
#-----------------------------------------------------------------------------------
if [ -n "$CENTRAL_BASE_IMAGE_URI" ]; then
  log "Base image: central ECR mirror (no NGC login)"
  BASE_IMAGE="$CENTRAL_BASE_IMAGE_URI"
  CENTRAL_REGISTRY="$(echo "$CENTRAL_BASE_IMAGE_URI" | cut -d/ -f1)"
  # A cross-account pull needs its own login; skip it when the mirror happens to live in
  # the registry we already authenticated against above.
  if [ "$CENTRAL_REGISTRY" != "$REGISTRY" ]; then
    log "Logging in to central ECR registry ${CENTRAL_REGISTRY}"
    aws ecr get-login-password --region "$AWS_DEFAULT_REGION" \
      | docker login --username AWS --password-stdin "$CENTRAL_REGISTRY"
  fi
elif [ -n "$NGC_SECRET_ID" ]; then
  log "Base image: nvcr.io (NGC login)"
  BASE_IMAGE="nvcr.io/nvidia/pytorch:25.04-py3"
  # Read the key into a variable and pipe it to --password-stdin. Never echo it, never
  # pass it as a docker flag (it would land in the process table), never put it in a log
  # line. The variable is unset again the moment the login is done.
  NGC_KEY="$(aws secretsmanager get-secret-value --secret-id "$NGC_SECRET_ID" \
    --region "$AWS_DEFAULT_REGION" --query SecretString --output text)"
  printf '%s' "$NGC_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
  unset NGC_KEY
else
  log "Base image: nvcr.io (no NGC_SECRET_ID; NGC login skipped, pulling anonymously)"
  BASE_IMAGE="nvcr.io/nvidia/pytorch:25.04-py3"
fi
log "Base image: ${BASE_IMAGE}"

#-----------------------------------------------------------------------------------
# 4. Build and push.
#
# Two tags and no more: :n17 is what the labs and job definitions reference, and the
# immutable dated :n17-YYYYMMDD pins exactly what an account received. A floating
# "latest" tag is deliberately NOT created or pushed - it makes it impossible to tell
# which build a participant account is actually running.
#-----------------------------------------------------------------------------------
DATE_TAG="n17-$(date -u +%Y%m%d)"
log "Building GR00T N1.7 training image (expect ~20 min, ~22.6 GB)"
# cd first so the -f path resolves: CodeBuild's working directory is not the repo root.
cd "$REPO_ROOT"
docker build -f containers/gr00t-training/Dockerfile.n17 \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -t "$ECR_REPO_URI:n17" -t "$ECR_REPO_URI:$DATE_TAG" "$REPO_ROOT"

log "Pushing ${ECR_REPO_URI}:n17"
docker push "$ECR_REPO_URI:n17"
log "Pushing ${ECR_REPO_URI}:${DATE_TAG}"
docker push "$ECR_REPO_URI:$DATE_TAG"

log "Done. Pushed ${ECR_REPO_URI}:n17 and ${ECR_REPO_URI}:${DATE_TAG}"
