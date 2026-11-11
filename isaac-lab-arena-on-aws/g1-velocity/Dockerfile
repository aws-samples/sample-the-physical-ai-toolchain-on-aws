# Isaac Lab Arena release/0.3.0 plus the workshop's G1 velocity-tracking evaluation.
#
# BASE_IMAGE is NVIDIA's own Arena image, built unchanged from the Arena repository's
# docker/Dockerfile.isaaclab_arena at release/0.3.0 (see event-build-lab-images.sh).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    PYTHONUNBUFFERED=1

COPY arena_g1_velocity.py arena_run.py arena_g1_velocity_eval.py g1_velocity_experiment.yaml /workspace/g1-velocity/

# AWS Batch runs this with no command override:
#   DATA_BUCKET, CHECKPOINT_PREFIX [, NUM_EPISODES, NUM_ENVS, RESULTS_PREFIX, REGION]
# Arena's own entrypoint creates a host user from DOCKER_RUN_* variables for interactive
# use; a Batch job runs as root instead, after the ldconfig step that entrypoint performs.
WORKDIR /workspace/g1-velocity
ENTRYPOINT []
CMD ["/bin/bash", "-c", "ldconfig && exec /isaac-sim/python.sh /workspace/g1-velocity/arena_g1_velocity_eval.py"]
