# Pinned by digest so the tag cannot be re-pushed under us. Tag kept for readability.
# Verify the digest resolves to linux/amd64 (docker buildx imagetools inspect) if the build errors on arch.
FROM nvcr.io/nvidia/isaac-sim:6.0.0@sha256:a9d06fb72cc2b39b2bf1cacc002e9f7cd1a5bbe3c3cfbbcf2e6de3a1d726d101

ENV ACCEPT_EULA=Y

# Switch to root for system installs
USER root

# Install AWS CLI v2 as standalone binary
RUN apt-get update && apt-get install -y --no-install-recommends unzip curl && \
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip && \
    unzip -q /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/aws /tmp/awscliv2.zip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies into Isaac Sim's Python
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    pyyaml==6.0.2 \
    numpy==1.26.4

# Create non-root user and grant access to Isaac Sim runtime
RUN useradd -m -s /bin/bash simuser && \
    mkdir -p /app /tmp/assets /tmp/results && \
    chown -R simuser:simuser /app /tmp/assets /tmp/results && \
    chmod +x /isaac-sim/python.sh && \
    chmod -R o+rX /isaac-sim/ && \
    # Renderer needs write access to shader cache and data directories.
    # Owned by the runtime user (simuser) with 755 instead of world-writable 777.
    mkdir -p /isaac-sim/kit/cache /isaac-sim/kit/data/documents && \
    chown -R simuser:simuser /isaac-sim/kit/cache /isaac-sim/kit/data/documents && \
    chmod -R 755 /isaac-sim/kit/cache /isaac-sim/kit/data/documents

COPY --chown=simuser:simuser simulation/ /app/simulation/
COPY --chown=simuser:simuser config/ /app/config/
COPY --chmod=755 --chown=simuser:simuser container/entrypoint.sh /app/entrypoint.sh

# Switch to non-root user for runtime
USER simuser

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
