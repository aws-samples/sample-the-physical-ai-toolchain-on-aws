"""Resolve explicit user choices into the recipe both execution modes consume.

This module is offline: no SageMaker imports, credential lookup or AWS calls.
"""
from __future__ import annotations

import json
import copy
import math
import re
import sys
from urllib.parse import urlparse

from .arena import gate_suite, resolve_runtime
from .registry import (
    DATASET_FROM_SUITE_MANIFEST,
    named_cells,
    resolve,
    resolve_suite,
    select_instance,
    select_volume_gb,
)
from .workflow import selected_steps
from .cli_context import command_name

GPU_STEPS = ("FineTune", "SimEval")


def assignments(values, label, convert=str):
    result = {}
    for value in values or []:
        name, separator, raw = value.partition("=")
        if not separator or name not in GPU_STEPS or not raw or name in result:
            raise ValueError(
                f"{label}: use unique FineTune=VALUE or SimEval=VALUE assignments; got {value!r}"
            )
        result[name] = convert(raw)
    return result


def selected_cell(args):
    cells = named_cells()
    explicit = (args.model, args.model_version, args.simulator, args.suite)
    if args.cell:
        if any(value is not None for value in explicit):
            raise ValueError("Use --cell or explicit model/version/simulator/suite, not both")
        if args.cell not in cells:
            raise ValueError(f"Unknown cell {args.cell!r}. Available: {', '.join(cells)}")
        return args.cell, cells[args.cell]
    if not args.model or not args.simulator or not args.suite:
        raise ValueError("Choose --cell, or supply --model, --simulator and --suite")
    if args.model == "gr00t" and args.model_version not in {"n16", "n17"}:
        raise ValueError("GR00T requires --model-version n16 or n17")
    if args.model != "gr00t" and args.model_version is not None:
        raise ValueError("--model-version currently applies only to gr00t")
    for name, cell in cells.items():
        if explicit == (cell["model"], cell["version"], cell["simulator"], cell["suite"]):
            return name, cell
    raise ValueError(
        f"This selection is not exposed by the current frontend. Run '{command_name()} cells' for "
        "available choices; the README lists additional legacy backend workflows."
    )


def resolve_run(args):
    args = copy.copy(args)
    name, cell = selected_cell(args)
    through = args.through or ("RegisterModel" if args.mode == "managed" else "SuccessGate")
    if args.mode == "local" and through == "RegisterModel":
        raise ValueError("RegisterModel requires managed execution")
    steps = selected_steps(through, checkpoint=bool(args.checkpoint_s3))
    if getattr(args, "sample", False):
        for step, field, value in (("FineTune", "train_steps", 200),
                                   ("SimEval", "eval_trials", 3),
                                   ("SuccessGate", "threshold", 0.0)):
            if step in steps and getattr(args, field) is None:
                setattr(args, field, value)
    if args.mode == "local" and not cell["local_profile"]:
        raise ValueError(f"{name} currently supports managed execution only")
    recording = getattr(args, "record_video", False)
    if recording == "auto":
        recording = cell["simulator"] == "isaac_arena" and "SimEval" in steps
    if recording and (cell["simulator"] != "isaac_arena" or "SimEval" not in steps):
        raise ValueError("--record-video requires an Arena run that includes SimEval")
    instances = assignments(args.instance, "--instance")
    volumes = assignments(args.volume_gb, "--volume-gb", int)
    images = assignments(args.image, "--image")
    if args.mode == "local" and instances:
        raise ValueError("Local execution uses the configured GPU host; omit --instance")
    for label, values in (("--instance", instances), ("--volume-gb", volumes), ("--image", images)):
        omitted = set(values) - set(steps)
        if omitted:
            raise ValueError(f"{label} selects omitted steps: {', '.join(sorted(omitted))}")
    required = [(True, "max_runtime_seconds"), ("FineTune" in steps, "train_steps"),
                ("SimEval" in steps, "eval_trials"), ("SuccessGate" in steps, "threshold")]
    missing = ["--" + field.replace("_", "-") for needed, field in required
               if needed and getattr(args, field) is None]
    if args.mode == "managed":
        missing.extend(f"--instance {step}=TYPE" for step in GPU_STEPS
                       if step in steps and step not in instances)
    if missing:
        raise ValueError("Required launch choices: " + ", ".join(missing)
                         + f". See {command_name()} run --help and {command_name()} cells --details.")
    if "FineTune" not in steps and (args.train_steps is not None or args.save_steps is not None):
        raise ValueError("A supplied checkpoint omits training; omit training-only arguments")
    if "SimEval" not in steps and (args.eval_trials is not None or args.eval_seed is not None):
        raise ValueError("Evaluation arguments were supplied but SimEval is omitted")
    if "SuccessGate" not in steps and args.threshold is not None:
        raise ValueError("--threshold requires SuccessGate")
    for label, value in (("train steps", args.train_steps), ("trials", args.eval_trials),
                         ("runtime", args.max_runtime_seconds), ("save steps", args.save_steps),
                         *[(f"{step} volume", value) for step, value in volumes.items()]):
        if value is not None and (isinstance(value, bool) or value <= 0):
            raise ValueError(f"{label} must be positive")
    if args.threshold is not None and (
        not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1
    ):
        raise ValueError("--threshold must be a finite success rate from 0 to 1")
    local_arguments = {
        "image_pull_timeout_seconds": args.image_pull_timeout_seconds,
        "container_preparation_seconds": args.container_preparation_seconds,
        "local_timeout_seconds": args.local_timeout_seconds,
    }
    for label, value in local_arguments.items():
        if value is not None and (args.mode != "local" or value <= 0):
            raise ValueError(f"--{label.replace('_', '-')} requires local mode and a positive value")
    if args.checkpoint_s3:
        uri = urlparse(args.checkpoint_s3)
        if uri.scheme != "s3" or not uri.netloc or not uri.path.strip("/") or uri.query or uri.fragment:
            raise ValueError("--checkpoint-s3 requires an exact s3://bucket/key object URI")
    if any(not re.fullmatch(r"ml\.[a-z0-9]+\.[a-z0-9]+", item) for item in instances.values()):
        raise ValueError("Managed instance types must look like ml.g6e.xlarge")
    if instances:
        from botocore.session import Session
        model = Session().get_service_model("sagemaker")
        accepted = model.operation_model("CreateTrainingJob").input_shape.members[
            "ResourceConfig"].members["InstanceType"].enum
        unknown = set(instances.values()) - set(accepted)
        if unknown:
            raise ValueError("The installed SageMaker API does not accept these training instances: "
                             + ", ".join(sorted(unknown))
                             + f". See {command_name()} cells --details for recommendations.")
    if getattr(args, "group", None):
        from .operations import validate_name
        validate_name(args.group)
    spec = resolve(cell["model"], cell["simulator"])
    suite = resolve_suite(cell["suite"])
    if cell["simulator"] == "isaac_arena":
        gate_suite(suite, allow_experimental=False)
    if cell["model"] == "gr00t" and "FineTune" in steps and suite.dataset is None:
        raise ValueError("This suite has no GR00T training dataset; provide --checkpoint-s3")
    if cell["model"] == "openvla" and (args.eval_trials or 0) > 50:
        raise ValueError("OpenVLA evaluation supports at most 50 trials per task")
    params = {
        "ModelFamily": cell["model"], "Suite": suite.name,
        "TrainSuite": "unified" if cell["model"] == "molmoact2" else suite.name,
        "ModelPackageGroupName": spec.registry_group,
        "DatasetS3Uri": DATASET_FROM_SUITE_MANIFEST,
        "TrainSaveSteps": args.save_steps or 1000000,
        "EvalTaskIds": "all", "EvalDoseSteps": "all",
        "EvalSeed": args.eval_seed if args.eval_seed is not None else (
            100 if cell["simulator"] == "isaac_arena" else 1000),
        "MaxRuntimeSeconds": args.max_runtime_seconds,
        "Gr00tVersion": cell["version"] or "n17",
        "UseGrootServer": spec.use_groot_server, "ArenaConnector": spec.arena_connector,
        # Diagnostic rollout capture, off unless --record-video. Sent as the string the
        # pipeline parameter's enum declares, not a Python bool.
        "EvalRecordVideo": "true" if getattr(args, "record_video", False) else "false",
        "VolumeSizeInGB": select_volume_gb(spec, "train", volumes.get("FineTune")),
        "EvalVolumeSizeInGB": select_volume_gb(spec, "eval", volumes.get("SimEval")),
    }
    for key, value in (("TrainSteps", args.train_steps), ("EvalTrials", args.eval_trials),
                       ("SuccessThreshold", args.threshold), ("CheckpointS3Uri", args.checkpoint_s3)):
        if value is not None:
            params[key] = value
    if suite.dataset and suite.dataset.revision:
        params["DatasetRevision"] = suite.dataset.revision
    if cell["version"] == "n16" and (args.train_steps or 0) > params["TrainSaveSteps"]:
        raise ValueError("N1.6 requires --save-steps >= --train-steps (final checkpoint only)")
    for step, role in (("FineTune", "train"), ("SimEval", "eval")):
        if step in steps:
            declared = spec.train_instance if role == "train" else spec.eval_instance
            if args.mode == "managed" and instances[step] != declared:
                print(f"{step}: selected {instances[step]}; this cell declares {declared}. "
                      "Declarations are not measured minima; see the README hardware evidence.",
                      file=sys.stderr)
            params["TrainInstanceType" if role == "train" else "EvalInstanceType"] = (
                "local_gpu" if args.mode == "local"
                else select_instance(spec, role, instances[step], acknowledged=True)
            )
    if cell["simulator"] == "isaac_arena":
        knobs = resolve_runtime(suite, cell["model"], cell["version"])
        params["EvalSimConfig"] = json.dumps(knobs, sort_keys=True)
        if recording:
            params["EvalRecordVideo"] = "true"
        for field, key in (
            ("ExpectedEmbodimentTag", "embodiment_tag"), ("ExpectedArenaObject", "object"),
            ("ExpectedArenaEmbodiment", "arena_embodiment"), ("ExpectedPolicyConfig", "policy_config_yaml"),
        ):
            params[field] = knobs[key]
            if len(params[field]) > 256:
                raise ValueError(f"{field} exceeds the managed processing environment limit")
    if args.checkpoint_s3:
        for key in ("TrainSteps", "TrainSaveSteps", "TrainImageUri", "TrainSourceDirUri",
                    "TrainInstanceType", "DatasetS3Uri", "TrainSuite"):
            params.pop(key, None)
    local_limits = {}
    if args.mode == "local":
        local_limits = {
            "image_pull_timeout_seconds": args.image_pull_timeout_seconds or 7200,
            "container_preparation_seconds": args.container_preparation_seconds or 600,
        }
        local_limits["total_timeout_seconds"] = args.local_timeout_seconds or (
            len(set(steps) & set(GPU_STEPS)) * args.max_runtime_seconds
            + 3 * local_limits["image_pull_timeout_seconds"]
            + local_limits["container_preparation_seconds"] + 3600
        )
        if local_limits["total_timeout_seconds"] >= 1000000:
            raise ValueError("Local total runtime must be below 1000000 seconds")
    return {"schema_version": 1, "cell": name, "selection": cell, "mode": args.mode,
            "steps": steps, "parameters": params, "image_overrides": images,
            "checkpoint_s3": args.checkpoint_s3, "local_limits": local_limits,
            "group": getattr(args, "group", None), "sample": bool(getattr(args, "sample", False)),
            "configuration": getattr(args, "configuration", {})}


def local_parameters(request, declarations):
    """Fill SDK secondary defaults; explicit user budgets remain in the request."""
    if request["mode"] != "local" or request["steps"] != selected_steps(
        request["steps"][-1], checkpoint=bool(request.get("checkpoint_s3"))
    ) or "RegisterModel" in request["steps"]:
        raise ValueError("Local execution requires a supported prefix ending before RegisterModel")
    cell = named_cells()[request["cell"]]
    if request["selection"] != cell:
        raise ValueError("Saved cell definition differs from this checkout; resolve a new request")
    values = {parameter.name: parameter.default_value for parameter in declarations.values()
              if parameter.default_value is not None}
    declared = {parameter.name for parameter in declarations.values()}
    values.update({key: value for key, value in request["parameters"].items() if key in declared})
    for key in ("TrainSourceDirUri", "EvalSourceDirUri"):
        if key in declared:
            values[key] = "__STAGED_LOCAL_SOURCE__"
    missing = {parameter.name for parameter in declarations.values()} - values.keys()
    if missing:
        raise ValueError(f"Resolved request lacks required pipeline parameters: {sorted(missing)}")
    return values
