"""User-visible launch choices: configuration, samples and explicit checkpoints."""
import json

import pytest

from vla_pipeline.cli import parser
from vla_pipeline.cli_settings import apply_settings
from vla_pipeline.launch_request import resolve_run
from vla_pipeline.registry import named_cells


def arguments(cell="gr00t-n16-arena", mode="local", *extra):
    values = ["run", "--deployment", "example", "--cell", cell, "--mode", mode,
              "--max-runtime-seconds", "21600"]
    if mode == "managed":
        values += ["--instance", "FineTune=ml.g6e.12xlarge", "--instance", "SimEval=ml.g6e.12xlarge"]
    return parser().parse_args(values + list(extra))


@pytest.mark.parametrize("cell,mode", [
    (name, mode) for name, cell in named_cells().items() for mode in ("local", "managed")
    if mode == "managed" or cell["local_profile"]
])
def test_sample_runs_the_same_workflow_as_explicit_budgets(cell, mode):
    sample = resolve_run(arguments(cell, mode, "--sample"))
    explicit = resolve_run(arguments(
        cell, mode, "--train-steps", "200", "--eval-trials", "3", "--threshold", "0"))
    assert sample["steps"] == explicit["steps"]
    assert sample["parameters"] == explicit["parameters"]
    assert "FineTune" in sample["steps"]


def test_explicit_budgets_override_sample_defaults():
    request = resolve_run(arguments(
        "gr00t-n16-arena", "local", "--sample", "--train-steps", "10", "--eval-trials", "1",
        "--threshold", "0.25"))
    assert request["parameters"]["TrainSteps"] == 10
    assert request["parameters"]["EvalTrials"] == 1
    assert request["parameters"]["SuccessThreshold"] == 0.25


def test_checkpoint_sample_does_not_train():
    request = resolve_run(arguments("gr00t-n16-arena", "local", "--sample",
                                    "--checkpoint-s3", "s3://example/model.tar.gz"))
    assert request["steps"] == ["SimEval", "Validate", "SuccessGate"]
    assert "TrainSteps" not in request["parameters"]
    assert request["checkpoint_s3"] == "s3://example/model.tar.gz"


def test_training_only_sample_does_not_request_evaluation_or_gate():
    request = resolve_run(arguments("gr00t-n16-arena", "local", "--sample", "--through", "FineTune"))
    assert request["steps"] == ["FineTune"]
    assert "EvalTrials" not in request["parameters"]
    assert "SuccessThreshold" not in request["parameters"]


def test_unsupported_local_cell_is_rejected():
    with pytest.raises(ValueError, match="managed execution only"):
        resolve_run(arguments("gr00t-n17-arena", "local", "--sample"))


@pytest.mark.parametrize("cell,extra", [
    ("molmoact2-libero", []), ("gr00t-n16-arena", ["--through", "FineTune"]),
])
def test_unsupported_recording_is_rejected(cell, extra):
    with pytest.raises(ValueError, match="requires an Arena run that includes SimEval"):
        resolve_run(arguments(cell, "local", "--sample", "--record-video", *extra))


@pytest.mark.parametrize("account_id", ["111122223333", 111122223333])
def test_shared_config_supplies_defaults_and_explicit_flags_win(tmp_path, account_id):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"arena": {
        "account_id": account_id, "deployment_name": "configured",
        "managed": {"cell": "gr00t-n16-arena", "train_instance_type": "ml.g6e.12xlarge",
                    "eval_instance_type": "ml.g6e.12xlarge"},
        "run": {"max_runtime_seconds": 21600, "record_video": "auto"},
    }}))
    args = parser().parse_args([
        "--config", str(config), "run", "--mode", "managed", "--sample", "--max-runtime-seconds", "3600",
        "--checkpoint-s3", "s3://example/model.tar.gz", "--instance", "SimEval=ml.g6e.8xlarge",
    ])
    apply_settings(args)
    assert args.deployment == "configured"
    assert args.configured_account == "111122223333"
    request = resolve_run(args)
    assert request["parameters"]["MaxRuntimeSeconds"] == 3600
    assert request["parameters"]["EvalInstanceType"] == "ml.g6e.8xlarge"
    assert request["parameters"]["EvalRecordVideo"] == "true"
    assert "TrainInstanceType" not in request["parameters"]


def test_full_run_requires_explicit_budgets_without_sample():
    with pytest.raises(ValueError, match="Required launch choices: --train-steps, --eval-trials, --threshold"):
        resolve_run(arguments())
