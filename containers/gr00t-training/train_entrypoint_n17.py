#!/usr/bin/env python3
"""GR00T N1.7-3B fine-tuning entrypoint for SageMaker and AWS Batch.

Runs inside the custom training container (see Dockerfile.n17). It:
  1. Reads SageMaker hyperparameters + channel/output paths
  2. Resolves the embodiment (NEW_EMBODIMENT) modality config to use
  3. Pre-flights one video decode so a codec problem fails in seconds
  4. Fine-tunes GR00T N1.7-3B via launch_finetune.py
  5. Saves the model + metadata to SM_MODEL_DIR (auto-uploaded to S3)

Embodiment selection (step 2) is driven by the environment, because the Docker
ENTRYPOINT dispatches on $1 and AWS Batch job definitions cannot override an
entrypoint — so the embodiment cannot be chosen from the outside any other way:

  MODALITY_CONFIG_PATH  explicit path to a modality config module. Wins when set.
  EMBODIMENT_CONFIG     'ur3' (default) | 'g1_dex3'. Named configs shipped at
                        /opt/ml/code/ by Dockerfile.n17.
  TUNE_VISUAL           '0' (default, matching upstream) | '1'. When off, the
                        vision tower stays frozen and only the projector + DiT
                        action head train, which is what makes a single 48 GB
                        L40S viable.

'ur3' keeps the original inlined config written to the output dir, so the
validated UR3 flow is unchanged when neither variable is set.

N1.7 changes from N1.6:
  - Model: nvidia/GR00T-N1.7-3B (Cosmos-Reason2-2B backbone)
  - Training: launch_finetune.py CLI (replaces experiment.run() API)
  - GPU: 48 GB+ VRAM required (L40S, A100, H100)
  - Action horizon: up to 40 steps
  - Python: 3.12
"""

import json
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path

# SageMaker conventions
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
FAILURE_FILE = "/opt/ml/output/failure"
HYPERPARAMS_FILE = "/opt/ml/input/config/hyperparameters.json"
SDK_DIR = os.environ.get("GROOT_SDK_DIR", "/workspace/gr00t-repo")

# Named embodiment configs shipped into the image by Dockerfile.n17. 'ur3' is
# handled separately (written inline, see write_embodiment_config) so the
# original UR3 behaviour is preserved exactly.
CODE_DIR = "/opt/ml/code"
EMBODIMENT_CONFIG_MODULES = {
    "ur3": None,  # inline, written to $OUTPUT_DIR/ur3_embodiment/ur3_config.py
    "g1_dex3": os.path.join(CODE_DIR, "g1_dex3_modality_config.py"),
}
# Both shipped configs call register_modality_config(..., NEW_EMBODIMENT).
EMBODIMENT_TAG = "new_embodiment"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"", "0", "false", "no", "off"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_failure(message: str) -> None:
    """Write the SageMaker failure file so the reason surfaces in describe-training-job."""
    try:
        os.makedirs(os.path.dirname(FAILURE_FILE), exist_ok=True)
        with open(FAILURE_FILE, "w") as f:
            f.write(message)
    except OSError:
        pass
    print(f"\n  FAILURE: {message}", file=sys.stderr, flush=True)


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean env var. An unrecognised value is a hard failure.

    Silently treating a typo ("TUNE_VISUAL=ture") as False would change which
    parameters train, and the only symptom would be a different memory profile
    many minutes in — so refuse instead of guessing.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    _write_failure(
        f"{name}='{raw}' is not a boolean. Use one of "
        f"{sorted(_TRUE_VALUES)} or {sorted(v for v in _FALSE_VALUES if v)}. "
        f"This job did NOT train a model."
    )
    sys.exit(1)


def _hp(name, default):
    """Read a SageMaker hyperparameter from the JSON file, with env fallback."""
    if os.path.exists(HYPERPARAMS_FILE):
        with open(HYPERPARAMS_FILE) as f:
            hps = json.load(f)
        if name in hps:
            return hps[name]
    return os.environ.get(f"SM_HP_{name}", os.environ.get(f"SM_HP_{name.upper()}", default))


def write_embodiment_config(output_dir: str) -> str:
    """Write the UR3 embodiment config for N1.7.

    N1.7 uses the same register_modality_config pattern as N1.6.
    The action space is EEF (end-effector from speedl teleop): arm RELATIVE + gripper ABSOLUTE.
    Action horizon increased to 40 (N1.7 maximum).
    """
    config_dir = os.path.join(output_dir, "ur3_embodiment")
    os.makedirs(config_dir, exist_ok=True)

    config_code = '''\
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat
from gr00t.data.embodiment_tags import EmbodimentTag

ur3_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROTVEC,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROTVEC,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(ur3_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''
    config_path = os.path.join(config_dir, "ur3_config.py")
    with open(config_path, "w") as f:
        f.write(config_code)
    return config_path


def resolve_embodiment_config(output_dir: str) -> str:
    """Resolve which modality config to pass to --modality-config-path.

    Precedence:
      1. MODALITY_CONFIG_PATH, if set. A value that does not exist is a hard
         failure, not a silent fallback: falling back to UR3 for joint-space
         data is exactly the crash this resolver exists to prevent.
      2. EMBODIMENT_CONFIG, defaulting to 'ur3' so an unset environment keeps
         the validated UR3 behaviour.

    Exits 1 via _write_failure() on an unknown or unusable value.
    """
    explicit = os.environ.get("MODALITY_CONFIG_PATH", "").strip()
    if explicit:
        if not os.path.exists(explicit):
            _write_failure(
                f"MODALITY_CONFIG_PATH is set to '{explicit}' but that file does not "
                f"exist in the container. Ship the config via Dockerfile.n17 or use "
                f"EMBODIMENT_CONFIG ({'|'.join(EMBODIMENT_CONFIG_MODULES)}) instead. "
                f"This job did NOT train a model."
            )
            sys.exit(1)
        return explicit

    name = os.environ.get("EMBODIMENT_CONFIG", "ur3").strip() or "ur3"
    if name not in EMBODIMENT_CONFIG_MODULES:
        _write_failure(
            f"Unknown EMBODIMENT_CONFIG '{name}'. Supported values: "
            f"{', '.join(sorted(EMBODIMENT_CONFIG_MODULES))}. Set MODALITY_CONFIG_PATH "
            f"to point at your own modality config module instead. "
            f"This job did NOT train a model."
        )
        sys.exit(1)

    module_path = EMBODIMENT_CONFIG_MODULES[name]
    if module_path is None:
        # UR3: written inline so the container stays self-contained.
        return write_embodiment_config(output_dir)

    if not os.path.exists(module_path):
        _write_failure(
            f"EMBODIMENT_CONFIG '{name}' maps to {module_path}, which is missing from "
            f"the image. Rebuild the container with buildspec-n17.yml so Dockerfile.n17 "
            f"COPYs it into {CODE_DIR}/. This job did NOT train a model."
        )
        sys.exit(1)
    return module_path


def preflight_video_decode(dataset_path: str) -> None:
    """Decode one frame from the first video in the dataset and print the codec.

    GR00T decodes with torchcodec (gr00t/utils/video_utils.get_frames_by_indices
    -> torchcodec.decoders.VideoDecoder) against the ffmpeg runtime in the image.
    The G1 Edu pick-and-place capture declares video.codec 'av1'. If that codec
    is not decodable here, the dataloader would only fail many minutes into the
    run, buried in worker tracebacks — so prove it up front instead.
    """
    videos_root = Path(dataset_path) / "videos"
    if not videos_root.is_dir():
        print(f"  Video pre-flight: skipped, no {videos_root}", flush=True)
        return

    sample = next(iter(sorted(videos_root.rglob("*.mp4"))), None)
    if sample is None:
        print(f"  Video pre-flight: skipped, no .mp4 under {videos_root}", flush=True)
        return

    try:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(
            str(sample), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=0
        )
        codec = getattr(decoder.metadata, "codec", "unknown")
        frame = decoder.get_frames_at(indices=[0]).data
        print(
            f"  Video pre-flight: OK — {sample.name} codec={codec} "
            f"frame={tuple(frame.shape)}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001 - any decode problem is fatal here
        _write_failure(
            f"Video decode pre-flight failed on {sample} ({type(e).__name__}: {e}). "
            f"This dataset's videos cannot be decoded by torchcodec/ffmpeg in this "
            f"image. The G1 Edu capture is AV1 (meta/info.json video.codec 'av1'): if "
            f"AV1 decode is unsupported here, transcode the videos to H.264 and "
            f"re-stage the dataset to S3, then resubmit. This job did NOT train a model."
        )
        sys.exit(1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run_training() -> None:
    """Fine-tune GR00T N1.7 via launch_finetune.py CLI."""
    import torch

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    base_model = _hp("base_model", "nvidia/GR00T-N1.7-3B")
    dataset_path = CHANNEL_TRAINING
    output_dir = MODEL_DIR
    max_steps = int(_hp("max_steps", "10000"))
    # batch_size is the GLOBAL, pre-accumulation batch: the trainer derives
    # per_device = global_batch_size // num_gpus (experiment.py:263-266), so on a
    # 1-GPU instance this value IS the per-device batch. DeepSpeed ZeRO only
    # engages at num_gpus>1 (experiment.py:258-262); a single-GPU run relies on a
    # frozen vision tower instead (TUNE_VISUAL=0, the default) to fit in 48 GB.
    global_batch_size = int(_hp("batch_size", "8"))
    learning_rate = float(_hp("learning_rate", "1e-4"))
    gradient_accumulation_steps = int(_hp("gradient_accumulation_steps", "2"))
    save_steps = int(_hp("save_steps", "2000"))
    # Vision-tower tuning is OFF by default, matching upstream
    # (finetune_config.py:52 and gr00t_n1d7.py:46 both default tune_visual=False).
    # Frozen, the trainable set is the projector + DiT action head only, and the
    # backbone retains no activation graph. Enabled, it adds ~1.15B trainable
    # parameters (and their fp32 grads + AdamW moments) plus the whole ViT/LLM
    # activation graph, which is why the 4-GPU configuration needs DeepSpeed ZeRO.
    tune_visual = _env_flag("TUNE_VISUAL", False)

    print("=== GR00T N1.7-3B Fine-Tuning ===")
    print(f"  Base model:       {base_model}")
    print(f"  Dataset:          {dataset_path}")
    print(f"  Output:           {output_dir}")
    print(f"  Max steps:        {max_steps}")
    print(f"  Global batch:     {global_batch_size}")
    print(f"  Learning rate:    {learning_rate}")
    print(f"  Grad accum:       {gradient_accumulation_steps}")
    print(f"  GPUs:             {num_gpus}")
    print(f"  Per-device batch: {global_batch_size // max(num_gpus, 1)}")
    print(f"  Embodiment:       {os.environ.get('EMBODIMENT_CONFIG', 'ur3')}")
    print(f"  Tune visual:      {tune_visual} (TUNE_VISUAL)")
    print(flush=True)

    # Resolve the embodiment config (inline UR3 by default, or a shipped module)
    config_path = resolve_embodiment_config(output_dir)
    print(f"  Embodiment config: {config_path}", flush=True)
    print(f"  Embodiment tag:    {EMBODIMENT_TAG}", flush=True)

    # Fail fast on an undecodable video codec rather than deep inside the dataloader
    preflight_video_decode(dataset_path)

    # Delete pre-computed stats (format may differ between SDK versions)
    import glob
    for stats_file in glob.glob(os.path.join(dataset_path, "meta", "*.stats.json")) + \
                      [os.path.join(dataset_path, "meta", "stats.json"),
                       os.path.join(dataset_path, "meta", "relative_stats.json")]:
        if os.path.exists(stats_file):
            os.remove(stats_file)
            print(f"  Deleted stale stats: {stats_file}", flush=True)

    # N1.7 uses launch_finetune.py CLI with torchrun internally.
    # NOTE: this CLI has no --gradient-checkpointing flag (verified via --help on
    # the built image), and the custom modules never call a checkpoint function
    # anyway. Memory is controlled via global-batch-size / grad-accum and by which
    # modules are tuned — --tune-visual is the expensive one, see TUNE_VISUAL.
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(num_gpus),
        "--standalone",
        os.path.join(SDK_DIR, "gr00t", "experiment", "launch_finetune.py"),
        "--base-model-path", base_model,
        "--dataset-path", dataset_path,
        "--output-dir", output_dir,
        "--modality-config-path", config_path,
        "--embodiment-tag", EMBODIMENT_TAG,
        "--global-batch-size", str(global_batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--max-steps", str(max_steps),
        "--num-gpus", str(num_gpus),
        "--save-steps", str(save_steps),
        "--save-total-limit", "3",
        "--no-tune-llm",
        "--tune-visual" if tune_visual else "--no-tune-visual",
        "--tune-projector",
        "--tune-diffusion-model",
        "--dataloader-num-workers", "4",
        "--color-jitter-params", "brightness", "0.3", "contrast", "0.4",
        "saturation", "0.5", "hue", "0.08",
    ]

    print(f"\n  Command: {shlex.join(cmd)}", flush=True)
    print(flush=True)

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"launch_finetune.py exited with code {result.returncode}")

    print(f"\n  Fine-tuning complete. Checkpoint saved to: {output_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def require_sdk():
    """Fail loud if the SDK isn't in the image."""
    if not (Path(SDK_DIR) / "gr00t").exists():
        _write_failure(
            f"Isaac-GR00T SDK not found at {SDK_DIR}/gr00t. The container was not built "
            f"with the SDK. This job did NOT train a model."
        )
        sys.exit(1)
    try:
        import torch  # noqa: F401
    except Exception as e:
        _write_failure(f"Container broken: 'import torch' failed ({e}).")
        sys.exit(1)


def main():
    try:
        run_training()
    except Exception as e:
        _write_failure(
            f"GR00T N1.7 fine-tune failed: {e}. If this was CUDA OOM: with "
            f"TUNE_VISUAL=1 the trainable set is ~2B parameters, which needs "
            f"DeepSpeed ZeRO and therefore num_gpus>1 (4x L40S). With the default "
            f"TUNE_VISUAL=0 the vision tower stays frozen and the job is expected "
            f"to fit one 48 GB L40S — lower BATCH_SIZE (it is the per-device batch "
            f"on a 1-GPU instance) and raise GRAD_ACCUM_STEPS to keep the effective "
            f"batch. Check CloudWatch logs for details."
        )
        traceback.print_exc()
        sys.exit(1)

    # Write metadata
    metadata = {
        "base_model": _hp("base_model", "nvidia/GR00T-N1.7-3B"),
        "max_steps": int(_hp("max_steps", "10000")),
        "global_batch_size": int(_hp("batch_size", "12")),
        "learning_rate": float(_hp("learning_rate", "1e-4")),
        "framework": "isaac-groot-n1.7",
        "backbone": "Cosmos-Reason2-2B (Qwen3-VL)",
        "min_vram": "48 GB",
        "action_horizon": 16,
        "embodiment_config": os.environ.get("EMBODIMENT_CONFIG", "ur3"),
        "modality_config_path": os.environ.get("MODALITY_CONFIG_PATH", ""),
        "embodiment_tag": EMBODIMENT_TAG,
        "tune_visual": _env_flag("TUNE_VISUAL", False),
    }
    with open(Path(MODEL_DIR) / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("\n  Metadata saved. Done.", flush=True)


if __name__ == "__main__":
    require_sdk()
    main()
