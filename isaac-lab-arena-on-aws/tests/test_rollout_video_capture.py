"""Rollout video capture: Arena's own --video, forwarded and default-off.

Isaac Lab Arena records rollouts through gymnasium.wrappers.RecordVideo, wired in
isaaclab_arena/evaluation/policy_runner.py. This component owns no capture code --
only the decision to pass --video/--video_dir. These tests pin the three properties
that decision has to keep:

  1. It is OFF unless asked. --video switches Arena to render_mode="rgb_array", a
     different environment construction from the argv behind the accepted reference
     runs, so a default-on flip would silently change what "a result" means.
  2. Both launchers carry the same knob. arena.py exists because the two Arena
     launchers drifted; a capture flag reachable from one of them is the same defect.
  3. The flags precede the positional task. policy_runner uses a choices-positional,
     so a main optional emitted after it is mis-bound and argparse reports it
     "required" even when passed -- a prior SimEval failed exactly that way.
"""
from __future__ import annotations

import pathlib
import re

_REPO = pathlib.Path(__file__).resolve().parents[1]
_EVAL_ENTRY = _REPO / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py"


def test_pipeline_declares_the_parameter_and_defaults_it_off():
    from vla_pipeline.config import PipelineConfig
    from vla_pipeline.pipeline import build_pipeline

    cfg = PipelineConfig(
        account_id="000000000000", region="us-east-1",
        training_role_arn="arn:aws:iam::000000000000:role/train",
        workload_role_arn="arn:aws:iam::000000000000:role/wl",
        validation_role_arn="arn:aws:iam::000000000000:role/val",
        trust_bucket="trust-bucket", handoff_bucket="handoff-bucket",
        role_arn="arn:aws:iam::000000000000:role/r", bucket="b")
    definition = build_pipeline(cfg).definition()

    assert "EvalRecordVideo" in definition
    # Reaches the container: the baked eval_entry reads this env key, not the parameter.
    assert "EVAL_RECORD_VIDEO" in definition

    import json
    parsed = json.loads(definition)
    param = next(p for p in parsed["Parameters"] if p["Name"] == "EvalRecordVideo")
    assert param["DefaultValue"] == "false", (
        "capture must be opt-in: --video renders the scene, which is not the "
        "configuration the accepted reference runs measured")


def test_launch_request_sends_the_parameter_as_an_enum_string():
    """SageMaker requires a string enum; inspect the actual resolved payload."""
    from vla_pipeline.cli import parser
    from vla_pipeline.launch_request import resolve_run

    args = parser().parse_args([
        "run", "--deployment", "d", "--cell", "gr00t-n16-arena", "--mode", "local",
        "--sample", "--max-runtime-seconds", "21600", "--record-video"])
    assert resolve_run(args)["parameters"]["EvalRecordVideo"] == "true"


def test_recording_defaults_off_without_a_config_preference(tmp_path):
    from vla_pipeline.cli import parser
    from vla_pipeline.cli_settings import apply_settings
    from vla_pipeline.launch_request import resolve_run

    config = tmp_path / "config.json"
    config.write_text("{}")
    base = ["--config", str(config), "run", "--deployment", "d", "--cell", "gr00t-n16-arena",
            "--mode", "local", "--sample", "--max-runtime-seconds", "21600"]
    for flags, expected in (([], False), (["--record-video"], True), (["--no-record-video"], False)):
        args = parser().parse_args(base + flags)
        apply_settings(args)
        request = resolve_run(args)
        assert (request["parameters"].get("EvalRecordVideo", "false") == "true") is expected


def test_submit_simeval_carries_the_same_knob():
    """Both Arena launchers set EVAL_RECORD_VIDEO, so neither can drift from the other."""
    import importlib.util

    from vla_pipeline import registry as reg

    spec = importlib.util.spec_from_file_location(
        "submit_simeval_video", _REPO / "scripts/submit_simeval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import types
    knobs = {"policy_config_yaml": "/w/c.yaml", "embodiment_tag": "GR1",
             "task_name": "put_item_in_fridge_and_close_door",
             "arena_embodiment": "gr1_joint", "object": "NONE"}
    base = dict(model_family="gr00t", gr00t_version="n16", use_groot_server="true",
                arena_connector="groot", eval_seed="100", eval_trials="1",
                eval_task_ids="all", checkpoint_s3="s3://b/p/x/model.tar.gz",
                hf_secret="s", posctrl_n16="false", posctrl_repo="",
                posctrl_revision="")

    suite = reg.resolve_suite("arena_gr1_fridge")
    off = mod.build_environment(
        types.SimpleNamespace(**base, record_video="false"), suite, knobs)
    on = mod.build_environment(
        types.SimpleNamespace(**base, record_video="true"), suite, knobs)

    assert off["EVAL_RECORD_VIDEO"] == "false"
    assert on["EVAL_RECORD_VIDEO"] == "true"


def test_eval_entry_gates_the_flags_and_emits_them_before_the_positional_task():
    """Source-level because the argv is built inside a SimulationApp-bound path.

    Ordering is the point: policy_runner's task is a choices-positional, so
    --video/--video_dir emitted after it are mis-bound. Checking the source
    positions catches a reordering that no offline behavioural test would reach.
    """
    source = _EVAL_ENTRY.read_text()

    guard = source.index('os.environ.get("EVAL_RECORD_VIDEO"')
    video_flag = source.index('"--video", "--video_dir"')
    positional = source.index("cmd.append(task_name)")

    assert guard < video_flag, "the flags must sit inside the EVAL_RECORD_VIDEO guard"
    assert video_flag < positional, (
        "--video/--video_dir are main optionals and must precede the positional task; "
        "policy_runner mis-binds an optional that follows its choices-positional")

    # Default off at the read site too, so an unset env var never enables rendering.
    assert re.search(r'EVAL_RECORD_VIDEO",\s*"false"', source), (
        "an unset EVAL_RECORD_VIDEO must read as off")

    # Written under the SageMaker model dir, which is what makes the mp4 arrive in
    # the SimEval ModelArtifacts without any extra upload step.
    assert 'f"{MODEL_DIR}/videos"' in source
