"""pai groot — Lab 1 GR00T workflow: convert data, train, deploy, serve.

Each subcommand is a thin wrapper over the validated Lab 1 scripts under
training/gr00t/ — the CLI resolves Foundation stack outputs and forwards to the
same code the lab doc runs by hand, so there is one source of truth:

    convert   zarr teleop episodes      -> LeRobot v2 dataset  (local)
    launch    LeRobot dataset           -> SageMaker Pipeline run (registers to groot-models)
    runs      list recent pipeline executions
    deploy    a trained model.tar.gz    -> SageMaker real-time endpoint
    invoke    call a live endpoint with a wrist image + robot state
    delete    tear down the endpoint (+ its config + model)
    ingest    bring-your-own Zarr data: convert -> upload -> [train]
"""

import os
import subprocess
import sys
from pathlib import Path

import click

from pai import cfn, config, helpers

STACK_NAME = "PhysicalAi-dev-Foundation"
HF_SECRET_NAME = "physical-ai/hf-token"


def _hf_token_from_secret(region: str) -> str:
    """Read the optional HF token secret (Lab 0); "" if it's absent or unreadable."""
    import boto3
    try:
        sm = boto3.client("secretsmanager", region_name=region)
        return sm.get_secret_value(SecretId=HF_SECRET_NAME)["SecretString"].strip()
    except Exception:
        return ""


@click.group()
def groot():
    """Lab 1 — fine-tune and deploy a GR00T manipulation policy."""
    pass


def _ensure_repo_on_path():
    """Put the repo root on sys.path so `import training...` resolves.

    The training/ tree ships with the repo but is not part of the installed
    `pai` package, so commands that import training.gr00t.* must add the repo
    root first (mirrors rl.py, which does the same for training.scripts.*).
    """
    repo = str(config.REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _require_data_deps():
    """Abort with a clear message if the data-conversion deps are missing.

    The convert/ingest scripts need zarr + opencv + pandas + pyarrow. These are
    part of the base install, so this only fails if `pai` was installed without
    its dependencies — catch it here and point at the install, instead of letting
    the subprocess die on a raw `ModuleNotFoundError` traceback.
    """
    import importlib.util

    missing = [m for m in ("zarr", "cv2", "pandas", "pyarrow")
               if importlib.util.find_spec(m) is None]
    if missing:
        helpers.error("Missing data-conversion dependencies: " + ", ".join(missing))
        helpers.info("Reinstall `pai` with its dependencies:")
        helpers.info("  pip install -e .")
        raise click.Abort()


def _groot_script(name: str) -> str:
    """Absolute path to a training/gr00t/<name> script."""
    return str(config.REPO_ROOT / "training" / "gr00t" / name)


# ---------------------------------------------------------------------------
# convert — Zarr teleop episodes -> LeRobot v2 dataset (local, no AWS)
# ---------------------------------------------------------------------------


@groot.command()
@click.option("--episodes-dir", default="training/data/episodes/episodes",
              help="Directory of raw Zarr episode_* folders")
@click.option("--output-dir", default="training/data/ur3_lerobot_dataset",
              help="Destination LeRobot v2 dataset directory")
@click.option("--camera", default="wrist", help="Primary camera key in the Zarr images group")
@click.option("--dry-run", is_flag=True, help="Print the command without converting")
def convert(episodes_dir, output_dir, camera, dry_run):
    """Convert Zarr teleop episodes to a LeRobot v2 dataset (Lab 1 Step 2)."""
    script = _groot_script("convert_zarr_to_lerobot.py")
    cmd = [sys.executable, script,
           "--episodes-dir", episodes_dir,
           "--output-dir", output_dir,
           "--camera", camera]

    if dry_run:
        helpers.info("[dry-run] Would run:")
        helpers.info(f"  {' '.join(cmd)}")
        helpers.info("\n[dry-run] No data converted.")
        return

    src = Path(episodes_dir)
    if not src.is_dir():
        helpers.error(f"Episodes directory not found: {episodes_dir}")
        helpers.info("Pull and extract the bundled dataset first (Lab 1 Step 2b):")
        helpers.info("  git lfs pull && unzip -o training/data/ur3_episodes_001_027.zip "
                     "-d training/data/episodes")
        raise click.Abort()

    _require_data_deps()

    helpers.heading("Converting Zarr -> LeRobot v2")
    try:
        helpers.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        helpers.error(f"Conversion failed with exit code {e.returncode}")
        raise click.Abort()


# ---------------------------------------------------------------------------
# upload — sync the converted LeRobot dataset to S3 (Lab 1 Step 3)
# ---------------------------------------------------------------------------


@groot.command()
@click.option("--dataset-dir", default="training/data/ur3_lerobot_dataset",
              help="Local LeRobot dataset directory to upload")
@click.option("--prefix", default="groot-data/ur3",
              help="S3 prefix under the datasets bucket (uploads to <prefix>/dataset/)")
@click.option("--dry-run", is_flag=True, help="Resolve the bucket and print the sync; upload nothing")
def upload(dataset_dir, prefix, dry_run):
    """Upload the converted LeRobot dataset to S3 (Lab 1 Step 3).

    Resolves the datasets bucket from the Foundation stack, then runs
    `aws s3 sync` to s3://<bucket>/<prefix>/dataset/ — where `pai groot launch`
    expects to find it.
    """
    region = config.resolve_region()

    bucket = cfn.bucket(STACK_NAME, region)
    if not bucket:
        helpers.error("Could not resolve the datasets bucket from the Foundation stack.")
        helpers.info("Deploy it first: pai deploy foundation")
        raise click.Abort()

    s3_uri = f"s3://{bucket}/{prefix}/dataset/"
    cmd = ["aws", "s3", "sync", dataset_dir, s3_uri]

    if dry_run:
        helpers.info("[dry-run] Would run:")
        helpers.info(f"  {' '.join(cmd)}")
        helpers.info("\n[dry-run] No upload performed.")
        return

    src = Path(dataset_dir)
    if not src.is_dir():
        helpers.error(f"Dataset directory not found: {dataset_dir}")
        helpers.info("Convert the episodes first: pai groot convert")
        raise click.Abort()

    helpers.heading(f"Uploading dataset to {s3_uri}")
    try:
        helpers.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        helpers.error(f"Upload failed with exit code {e.returncode}")
        raise click.Abort()
    helpers.success("\nUpload complete. Launch training with: pai groot launch")


# ---------------------------------------------------------------------------
# launch — drive the SageMaker Pipeline (create-if-missing -> execute)
# ---------------------------------------------------------------------------


@groot.command()
@click.option("--dataset-prefix", default="groot-data/ur3",
              help="S3 prefix under the datasets bucket (expects <prefix>/dataset/)")
@click.option("--max-steps", type=int, default=5000, help="Training steps (100=smoke, 5000=full)")
@click.option("--batch-size", type=int, default=8, help="Per-GPU batch size")
@click.option("--instance-type", default="ml.g5.12xlarge", help="Training instance type")
@click.option("--dry-run", is_flag=True, help="Resolve config and preview; make no AWS calls")
def launch(dataset_prefix, max_steps, batch_size, instance_type, dry_run):
    """Run the GR00T fine-tuning Pipeline on SageMaker (Lab 1 Step 5).

    Creates the pipeline if it does not exist, then starts an execution that
    trains and registers the model to the `groot-models` registry. Upload the
    dataset to s3://<bucket>/<prefix>/dataset/ first (Lab 1 Step 3).
    """
    region = config.resolve_region()

    helpers.heading("GR00T Fine-Tuning Pipeline (SageMaker)")

    helpers.info("[1/2] Reading infrastructure outputs...")
    # Try SSM parameters first (Terraform/CFN bootstrap), fall back to CDK stack
    bucket = cfn.bucket(STACK_NAME, region)
    role_arn = cfn.role_arn(STACK_NAME, region)
    ecr_uri = cfn.ecr_uri(STACK_NAME, region, repo="gr00t-training")
    if not (bucket and role_arn and ecr_uri):
        helpers.error("Failed to read infrastructure outputs. Is the foundation deployed?")
        helpers.info(f"  Bucket:   {bucket or 'MISSING'}")
        helpers.info(f"  Role:     {role_arn or 'MISSING'}")
        helpers.info(f"  ECR:      {ecr_uri or 'MISSING'}")
        helpers.info("\nDeploy foundation first:")
        helpers.info("  aws cloudformation deploy --template-file workshop/bootstrap.cfn.yaml \\")
        helpers.info("    --stack-name physical-ai-foundation --capabilities CAPABILITY_NAMED_IAM")
        raise click.Abort()

    image = f"{ecr_uri}:latest"
    helpers.info(f"  Bucket:   {bucket}")
    helpers.info(f"  Role:     {role_arn}")
    helpers.info(f"  Image:    {image}")

    # HF_TOKEN is forwarded to the pipeline at execute time; never echo it.
    # The env var wins; otherwise fall back to the optional Secrets Manager secret
    # from Lab 0, so a new terminal doesn't silently lose the token.
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_source = "env"
    if not hf_token and not dry_run:
        hf_token = _hf_token_from_secret(region)
        hf_source = HF_SECRET_NAME
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token  # pipeline.execute_pipeline reads it from env
    if hf_token:
        helpers.info(f"  HF_TOKEN: <redacted> (from {hf_source}; passed to the pipeline at execute time)")
    elif dry_run:
        helpers.info(f"  HF_TOKEN: not in env; a real launch also checks {HF_SECRET_NAME}")
    else:
        helpers.warn(f"  HF_TOKEN: not set (env or {HF_SECRET_NAME}) — base-model download runs "
                     "unauthenticated (rate-limit-prone)")

    if dry_run:
        helpers.info("\n[2/2] [dry-run] Would create/update pipeline 'groot-finetune-pipeline' then execute:")
        helpers.info(f"  dataset:     s3://{bucket}/{dataset_prefix}/dataset/")
        helpers.info(f"  max-steps:   {max_steps}")
        helpers.info(f"  batch-size:  {batch_size}")
        helpers.info(f"  instance:    {instance_type}")
        helpers.info(f"  registry:    groot-models")
        helpers.info("\n[dry-run] No AWS calls made.")
        return

    _ensure_repo_on_path()
    from training.gr00t import pipeline

    helpers.info("\n[2/2] Creating/updating pipeline and starting execution...")
    try:
        pipeline.create_pipeline(
            s3_bucket=bucket, role_arn=role_arn, ecr_image=image,
            region=region, instance_type=instance_type,
        )
        result = pipeline.execute_pipeline(
            dataset_prefix=dataset_prefix, max_steps=max_steps,
            batch_size=batch_size, region=region,
        )
    except Exception as e:
        helpers.friendly_boto_error(e)
        return

    if result.get("status") == "error":
        helpers.error(result["message"])
        raise click.Abort()

    helpers.success("\nPipeline execution started.")
    helpers.info(f"  ARN: {result.get('execution_arn', '')}")
    helpers.info("\nMonitor with:")
    helpers.info("  pai groot runs")


# ---------------------------------------------------------------------------
# runs — list recent pipeline executions
# ---------------------------------------------------------------------------


@groot.command()
def runs():
    """List recent GR00T pipeline executions (Lab 1 Step 7b)."""
    region = config.resolve_region()

    _ensure_repo_on_path()
    from training.gr00t import pipeline

    try:
        result = pipeline.list_runs(region=region)
    except Exception as e:
        helpers.friendly_boto_error(e)
        return

    if result.get("status") == "error":
        helpers.error(result["message"])
        return

    helpers.heading(f"Pipeline: {result['pipeline']}")
    runs_list = result.get("runs", [])
    if not runs_list:
        helpers.info("No executions yet. Start one with: pai groot launch")
        return
    for r in runs_list:
        helpers.info(f"  {r['created']}  {r['status']:<12}  {r['arn']}")


# ---------------------------------------------------------------------------
# deploy / invoke / delete — SageMaker real-time endpoint (Lab 1 Step 9)
# ---------------------------------------------------------------------------


@groot.command()
@click.option("--model-s3", required=True, help="S3 URI of a training job's model.tar.gz")
@click.option("--endpoint-name", default="groot-ur3", help="Endpoint name to create")
@click.option("--instance-type", default="ml.g5.2xlarge", help="Endpoint instance type (needs a GPU)")
@click.option("--dry-run", is_flag=True, help="Show the API calls; make none")
def deploy(model_s3, endpoint_name, instance_type, dry_run):
    """Serve a fine-tuned model as a SageMaker real-time endpoint (Lab 1 Step 9).

    The endpoint runs the groot-inference container and loads model.tar.gz.
    GR00T loads slowly, so the startup health-check timeout is 30 minutes.
    """
    region = config.resolve_region()

    _ensure_repo_on_path()
    from training.gr00t import deploy_endpoint

    try:
        result = deploy_endpoint.deploy(
            model_s3=model_s3, endpoint_name=endpoint_name,
            instance_type=instance_type, region=region, dry_run=dry_run,
        )
    except Exception as e:
        helpers.friendly_boto_error(e)
        return

    if not dry_run:
        helpers.success(f"\nEndpoint '{endpoint_name}' is creating (~10-30 min to come InService).")
        helpers.info("  Call it:  pai groot invoke --endpoint-name "
                     f"{endpoint_name} --image-path wrist.jpg --task '...'")
        helpers.info(f"  Tear down: pai groot delete --endpoint-name {endpoint_name}")


@groot.command()
@click.option("--endpoint-name", default="groot-ur3", help="Endpoint to call")
@click.option("--image-path", required=True, help="Path to a wrist image file (jpg/png)")
@click.option("--state", default="0,0,0,0,0,0,0", help="Comma-separated 7D robot state")
@click.option("--task", default="", help="Task description, e.g. 'pick up the red cube'")
def invoke(endpoint_name, image_path, state, task):
    """Call a live endpoint with a wrist image + robot state -> action chunk."""
    region = config.resolve_region()

    if not Path(image_path).is_file():
        helpers.error(f"Image not found: {image_path}")
        raise click.Abort()

    _ensure_repo_on_path()
    from training.gr00t import deploy_endpoint

    try:
        deploy_endpoint.invoke(endpoint_name, image_path, state, task, region)
    except Exception as e:
        helpers.friendly_boto_error(e)


@groot.command()
@click.option("--endpoint-name", default="groot-ur3", help="Endpoint to delete")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def delete(endpoint_name, yes):
    """Delete the endpoint, its config, and its model (stops hourly billing)."""
    region = config.resolve_region()

    if not yes:
        if not click.confirm(f"Delete endpoint '{endpoint_name}' (+ its config and model)?"):
            helpers.info("Aborted.")
            return

    _ensure_repo_on_path()
    from training.gr00t import deploy_endpoint

    helpers.heading(f"Deleting endpoint {endpoint_name}")
    try:
        deploy_endpoint.delete(endpoint_name, region)
        helpers.success("Done.")
    except Exception as e:
        helpers.friendly_boto_error(e)


# ---------------------------------------------------------------------------
# ingest — bring-your-own Zarr data: convert -> upload -> [train]
# ---------------------------------------------------------------------------


@groot.command()
@click.option("--episodes-dir", required=True, help="Directory of your raw Zarr episode_* folders")
@click.option("--prefix", default="groot-data/custom", help="S3 prefix under the datasets bucket")
@click.option("--train", is_flag=True, help="Launch a training run after upload")
@click.option("--max-steps", type=int, default=100, help="Training steps if --train is set")
@click.option("--dry-run", is_flag=True, help="Show the plan; make no AWS calls and convert nothing")
def ingest(episodes_dir, prefix, train, max_steps, dry_run):
    """Bring your own data: convert Zarr -> upload to S3 -> optionally train.

    The expected Zarr schema is documented in docs/zarr-schema.md.
    """
    region = config.resolve_region()
    script = _groot_script("ingest_customer_data.py")
    cmd = [sys.executable, script,
           "--episodes-dir", episodes_dir, "--prefix", prefix,
           "--region", region, "--max-steps", str(max_steps)]
    if train:
        cmd.append("--train")
    if dry_run:
        cmd.append("--dry-run")

    if dry_run:
        helpers.info("[dry-run] Would run:")
        helpers.info(f"  {' '.join(cmd)}")

    if not dry_run:
        if not Path(episodes_dir).is_dir():
            helpers.error(f"Episodes directory not found: {episodes_dir}")
            raise click.Abort()
        _require_data_deps()

    helpers.heading("Ingesting custom data (convert -> upload" + (" -> train" if train else "") + ")")
    try:
        helpers.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        helpers.error(f"Ingestion failed with exit code {e.returncode}")
        raise click.Abort()


# ---------------------------------------------------------------------------
# record / control — physical UR3 hardware loop (optional, bring-your-own-robot)
# ---------------------------------------------------------------------------
#
# These two commands bracket the cloud pipeline with the physical robot: `record`
# captures teleop demonstrations into the Zarr layout `convert` reads, and
# `control` runs a deployed endpoint as a closed-loop policy on the arm. They need
# a real UR3 (+ wrist camera) reachable over the network — everything else in
# `pai groot` runs without hardware. See the optional bring-your-own-robot sections
# in workshop/lab-1-train-groot.md (Step 2 for record, Step 10 for control).


def _require_ur3_reachable(robot_ip: str):
    """Warn early if the UR3 URScript port isn't reachable at robot_ip:30002.

    Not fatal — the underlying tools also handle connection errors — but a clear
    up-front message beats a mid-run socket traceback.
    """
    import socket

    try:
        with socket.create_connection((robot_ip, 30002), timeout=2.0):
            return
    except OSError:
        helpers.warn(f"No UR3 responding at {robot_ip}:30002 (URScript port).")
        helpers.info("  Set the arm's IP with ROBOT_IP, e.g. ROBOT_IP=192.168.1.100 pai groot ...")
        helpers.info("  This command needs a physical UR3 + wrist camera on the network.")


@groot.command()
@click.option("--task", required=True, help="Task label for the episodes, e.g. 'pick up the red cube'")
@click.option("--mode", type=click.Choice(["gamepad", "keyboard"]), default="gamepad",
              help="gamepad: browser + game controller; keyboard: WASD (SSH-tunnel friendly)")
@click.option("--robot-ip", default=None, help="UR3 IP (default: $ROBOT_IP or 127.0.0.1 for a sim)")
@click.option("--port", type=int, default=8765, help="Local port for the browser teleop UI")
@click.option("--episodes-dir", default="training/data/episodes/episodes",
              help="Where to write Zarr episodes (what `pai groot convert` reads)")
@click.option("--dry-run", is_flag=True, help="Show what would run; start nothing")
def record(task, mode, robot_ip, port, episodes_dir, dry_run):
    """Teleoperate a physical UR3 and record demonstrations to Zarr (optional, needs hardware).

    Opens a browser teleop UI (gamepad) or reads the keyboard (SSH-friendly),
    drives the arm, and records synchronized joint/gripper/camera data into
    <episodes-dir>/episode_* — the exact layout `pai groot convert` consumes.
    Requires a UR3 + wrist camera on the network; not part of the cloud path.
    """
    robot_ip = robot_ip or os.environ.get("ROBOT_IP", "127.0.0.1")
    module = "robot.ur3.gamepad_teleop" if mode == "gamepad" else "robot.ur3.keyboard_teleop"
    cmd = [sys.executable, "-m", module, "--record", task, "--port", str(port)]

    if dry_run:
        helpers.info("[dry-run] Would run:")
        helpers.info(f"  ROBOT_IP={robot_ip} PAI_EPISODES_DIR={episodes_dir} {' '.join(cmd)}")
        helpers.info("\n[dry-run] No teleop session started.")
        return

    _require_data_deps()
    _require_ur3_reachable(robot_ip)

    env = os.environ.copy()
    env["ROBOT_IP"] = robot_ip
    env["PAI_EPISODES_DIR"] = str(Path(episodes_dir).resolve())

    helpers.heading(f"Teleop recording ({mode}) -> {episodes_dir}")
    helpers.info(f"  Robot: {robot_ip}   Task: {task}")
    if mode == "gamepad":
        helpers.info(f"  A browser window will open at http://localhost:{port}")
    helpers.info("  Press the record control (or Ctrl+C) to stop.\n")
    try:
        helpers.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        helpers.error(f"Teleop session exited with code {e.returncode}")
        raise click.Abort()
    helpers.success("\nRecording done. Next: pai groot convert")


@groot.command()
@click.option("--task", required=True, help="Natural-language task, e.g. 'pick up the red cube'")
@click.option("--endpoint-name", default="groot-ur3", help="Deployed endpoint to drive the arm")
@click.option("--robot-ip", default=None, help="UR3 IP (default: $ROBOT_IP or 127.0.0.1 for a sim)")
@click.option("--max-queries", type=int, default=20, help="Max endpoint queries before stopping")
@click.option("--save-images", is_flag=True, help="Save wrist frames to /tmp for debugging")
@click.option("--dry-run", is_flag=True, help="Show what would run; start nothing")
def control(task, endpoint_name, robot_ip, max_queries, save_images, dry_run):
    """Run a deployed endpoint as a closed-loop policy on a physical UR3 (optional, needs hardware).

    Captures the wrist camera, queries the SageMaker endpoint for an action
    chunk, and executes it on the arm at the control rate — the closing step of
    the loop. Deploy the endpoint first with `pai groot deploy`. Requires a UR3
    + wrist camera on the network.
    """
    region = config.resolve_region()
    robot_ip = robot_ip or os.environ.get("ROBOT_IP", "127.0.0.1")
    cmd = [sys.executable, "-m", "robot.ur3.control", task,
           "--endpoint", endpoint_name, "--max-queries", str(max_queries)]
    if save_images:
        cmd.append("--save-images")

    if dry_run:
        helpers.info("[dry-run] Would run:")
        helpers.info(f"  ROBOT_IP={robot_ip} AWS_DEFAULT_REGION={region} {' '.join(cmd)}")
        helpers.info("\n[dry-run] No control loop started.")
        return

    _require_data_deps()
    _require_ur3_reachable(robot_ip)

    env = os.environ.copy()
    env["ROBOT_IP"] = robot_ip
    env["AWS_DEFAULT_REGION"] = region

    helpers.heading(f"Closed-loop control -> {endpoint_name}")
    helpers.info(f"  Robot: {robot_ip}   Task: {task}   Region: {region}")
    helpers.warn("  The arm will MOVE. Keep the e-stop within reach.\n")
    try:
        helpers.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        helpers.error(f"Control loop exited with code {e.returncode}")
        raise click.Abort()


def register(cli: click.Group):
    """Register groot commands with the CLI."""
    cli.add_command(groot)
