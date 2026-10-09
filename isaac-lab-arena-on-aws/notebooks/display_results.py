"""Display an exported run report; this module never submits jobs or calls AWS.

The ``pai arena report`` command exports existing run/worker evidence.
This notebook presentation helper reads those files without substituting sample
numbers, the introductory demo, or another run's video.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path


def _read(directory, name, *, required=False):
    path = directory / name
    if not path.exists() and not required:
        return {}
    return json.loads(path.read_text())


def _text(value):
    return "Not reported" if value is None else str(value)


def _duration(value):
    if value is None:
        return "Not reported"
    minutes, seconds = divmod(round(float(value)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"


def summary_html(directory, *, expected_run_id):
    """Format saved evidence, keeping training and evaluation counts separate."""
    directory = Path(directory)
    run = _read(directory, "run.json", required=True)
    if run.get("id") != expected_run_id:
        raise ValueError("The report belongs to a different run; refusing stale results.")
    training = _read(directory, "training-summary.json")
    evaluation = _read(directory, "evaluation-summary.json")
    metrics = _read(directory, "metrics.json")
    params = run.get("request", {}).get("parameters", {})
    dataset = metrics.get("checkpoint_manifest", {}).get("dataset_manifest", {})
    episodes = evaluation.get("episodes")
    if episodes is not None and (type(episodes) is not int or episodes < 0):
        raise ValueError("The report contains an invalid evaluation episode count.")
    per_task = evaluation.get("per_task", [])
    successes = None
    if per_task and all(type(row.get("successes")) is int and type(row.get("episodes")) is int
                        for row in per_task):
        if any(row["episodes"] < 0 or not 0 <= row["successes"] <= row["episodes"] for row in per_task):
            raise ValueError("The report contains invalid task outcome counts.")
        if sum(row["episodes"] for row in per_task) != episodes:
            raise ValueError("Per-task counts disagree with the reported evaluation total.")
        successes = sum(row["successes"] for row in per_task)
    rate = evaluation.get("success_rate")
    if rate is not None and (type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1):
        raise ValueError("The report contains an invalid success rate.")
    if episodes and successes is not None and rate is not None:
        if not math.isclose(successes / episodes, rate, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("The reported success rate disagrees with the episode counts.")

    performed = training.get("training_performed")
    reused_checkpoint = bool(run.get("request", {}).get("checkpoint_s3") or params.get("CheckpointS3Uri"))
    if performed is False:
        training_description = (
            "No training in this run — reused an existing checkpoint."
            if reused_checkpoint else "Training was not performed in this run."
        )
    elif performed is True:
        steps = training.get("observed_final_step")
        training_description = (
            f"Fine-tuned the policy; observed final optimizer step: {_text(steps)}."
        )
    else:
        training_description = "Training completion has not been reported in the exported evidence."
    if episodes is None:
        evaluation_description = "Evaluation episode counts have not been reported."
    elif successes is None:
        evaluation_description = f"Evaluated {episodes} episodes; exact success count was not reported."
    else:
        evaluation_description = (
            f"The robot completed {successes} of {episodes} evaluation episodes successfully"
            + (f" ({100 * successes / episodes:.1f}%)." if episodes else ".")
        )

    execution = {
        "local": "GPU EC2 instance (--mode local)",
        "managed": "SageMaker managed jobs (--mode managed)",
    }.get(run.get("mode"), run.get("mode", "Not reported"))
    rows = [
        ("Run", run["id"]),
        ("Cell / execution", f"{run.get('cell', 'Not reported')} / {execution}"),
        ("Workflow status", run.get("status")),
        ("Independent verification", run.get("verification_status")),
        ("Training performed in this run", {True: "Yes", False: "No"}.get(performed)),
        ("Requested optimizer steps", training.get("requested_optimizer_steps")),
        ("Observed final optimizer step", training.get("observed_final_step")),
        ("Checkpoint's training dataset", dataset.get("source")),
        ("Demonstrations available in that dataset", dataset.get("episode_count")),
        ("Evaluation suite", evaluation.get("suite")),
        ("Evaluation episodes", episodes),
        ("Successful episodes", successes),
        ("Success rate", f"{100 * rate:.1f}%" if rate is not None else None),
        ("Evaluation seed", params.get("EvalSeed")),
        ("Success threshold", params.get("SuccessThreshold")),
        ("Elapsed run time", _duration(run.get("elapsed_seconds"))),
        ("Registered model package", run.get("model_package_arn", "Not reported / not requested")),
        ("Model approval status", run.get("model_approval_status")),
        ("Failure reason", run.get("failure_reason") or "None reported"),
    ]
    if performed is False:
        rows[5:7] = [("Training steps in this execution",
                     "Not applicable — checkpoint reuse" if reused_checkpoint else "Training not performed")]
    esc = lambda value: html.escape(_text(value))
    table = "".join(f"<tr><th style='text-align:left;padding:5px 16px 5px 0'>{esc(k)}</th>"
                    f"<td>{esc(v)}</td></tr>" for k, v in rows)
    tasks = ""
    if per_task:
        task_rows = "".join(
            f"<tr><td>{esc(row.get('task', row.get('task_id')))}</td>"
            f"<td>{esc(row.get('successes'))}</td><td>{esc(row.get('episodes'))}</td></tr>"
            for row in per_task)
        tasks = ("<h4>Task outcomes</h4><table><tr><th>Task</th><th>Successes</th>"
                 f"<th>Episodes</th></tr>{task_rows}</table>")
    interpretation = (
        "A zero-threshold sample can pass the workflow even with zero robot successes."
        if params.get("SuccessThreshold") == 0
        else "The workflow status and the robot's task success rate answer different questions."
    )
    return (
        f"<h3>What happened in this run</h3><p>{esc(training_description)} "
        f"{esc(evaluation_description)}</p><table>{table}</table>{tasks}"
        "<p>Dataset demonstrations are available training examples, not the number of evaluation "
        "episodes or proof that training consumed every demonstration.</p>"
        f"<p>{esc(interpretation)}</p>"
    )


def show_run_results(directory, *, expected_run_id):
    """Show the summary and only videos exported and identified for this run."""
    from IPython.display import HTML, Video, display

    directory = Path(directory).resolve()
    display(HTML(summary_html(directory, expected_run_id=expected_run_id)))
    video_index = _read(directory, "videos.json")
    if not video_index:
        run = _read(directory, "run.json", required=True)
        requested = run.get("request", {}).get("parameters", {}).get("EvalRecordVideo")
        if requested is True or str(requested).lower() == "true":
            raise ValueError("This run requested recording, but its video export record is missing.")
        print("Video: no video export record was provided. No demonstration clip is substituted.")
        return
    if video_index.get("run_id") != expected_run_id:
        raise ValueError("The video export belongs to a different run.")
    status = video_index.get("status")
    if status in {"not_supported", "not_requested", "not_exported"}:
        print("Video:", video_index.get("reason") or status.replace("_", " "))
        return
    videos = video_index.get("videos", [])
    if status != "recorded" or not videos:
        raise ValueError("Recording was requested but this run has no usable exported video.")
    for item in videos:
        path = (directory / item["path"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("A video path points outside this run's exported report.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
            raise ValueError("A downloaded video does not match its recorded SHA256.")
        print(f"Actual recording from {expected_run_id}: {path.name}")
        display(Video(filename=str(path), embed=True, width=800, html_attributes="controls"))
