"""Pre-flight checks the bridge runs BEFORE any compute is started.

Ported from the toolchain's own tooling so a run rejected here would also have been rejected (or
would have failed at startup) on the direct path:

* ``validate_dataset`` — ``training/gr00t/validate_lerobot_gr00t_dataset.py`` checks A (metadata
  files present and parseable), B/C/D (modality.json state/action groups: integer ranges, no
  ``original_key``, tiling ``[0, dims)``, matching names), E (video keys resolve to features),
  G (every episode's parquet and mp4 exist, every episode has ``length``) and the tasks half of
  I (tasks.jsonl yields at least one index). Checks H and the parquet half of I read parquet
  columns and are NOT ported: the Lambda has no pyarrow, and the container's own loader asserts
  them at startup.
* ``check_batch_divides_gpus`` — ``training/gr00t/submit_batch_training_n17.py``
  ``_check_batch_divides_gpus``: a GLOBAL batch the trainer cannot split across the job's GPUs.

Both operate on plain data (dicts, lists, a set of relative keys) so they need no AWS access and
can be exercised by the tests beside this file.
"""
from __future__ import annotations

import json
from typing import Iterable

# Loader defaults, from gr00t/data/dataset/lerobot_episode_loader.py.
FEATURE_FOR_MODALITY = {"state": "observation.state", "action": "action"}

META_FILES = ("meta/info.json", "meta/modality.json", "meta/episodes.jsonl", "meta/tasks.jsonl")

MAX_REPORTED_MISSING_EPISODES = 5


class DatasetInvalid(ValueError):
    """Raised with the full failure list; the message is what reaches SendTaskFailure."""

    def __init__(self, failures: list[str]):
        self.failures = failures
        shown = failures[:12]
        more = f" (+{len(failures) - 12} more)" if len(failures) > 12 else ""
        super().__init__("LeRobot dataset rejected: " + "; ".join(shown) + more)


def parse_json(text: str, name: str, failures: list[str]):
    try:
        return json.loads(text)
    except (TypeError, ValueError) as exc:
        failures.append(f"{name}: not valid JSON ({exc})")
        return None


def parse_jsonl(text: str, name: str, failures: list[str]):
    rows = []
    for n, line in enumerate((text or "").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            failures.append(f"{name}:{n}: not valid JSON ({exc})")
            return None
    return rows


def _check_groups(modality: dict, features: dict, failures: list[str]) -> dict:
    """Checks B, C and D."""
    validated: dict[str, dict] = {}
    for modality_type in ("state", "action"):
        section = modality.get(modality_type)
        if not isinstance(section, dict) or not section:
            failures.append(f"modality.json['{modality_type}'] is missing or empty")
            continue
        feature_key = FEATURE_FOR_MODALITY[modality_type]
        feature = features.get(feature_key)
        if not isinstance(feature, dict) or not feature.get("shape"):
            failures.append(f"info.json features['{feature_key}'] is missing or has no shape")
            continue
        declared = int(feature["shape"][0])

        groups: dict[str, dict] = {}
        shaped_ok = True
        for name, spec in section.items():
            if not isinstance(spec, dict):
                failures.append(f"modality.json['{modality_type}']['{name}'] is not an object")
                shaped_ok = False
                continue
            start, end = spec.get("start"), spec.get("end")
            if not isinstance(start, int) or isinstance(start, bool) \
                    or not isinstance(end, int) or isinstance(end, bool):
                failures.append(f"modality.json['{modality_type}']['{name}'] start/end must be integers")
                shaped_ok = False
                continue
            if end <= start:
                failures.append(f"modality.json['{modality_type}']['{name}'] has end <= start ({start}:{end})")
                shaped_ok = False
                continue
            if "original_key" in spec:  # check C
                failures.append(f"modality.json['{modality_type}']['{name}'] sets original_key; "
                                f"state/action groups must omit it")
                shaped_ok = False
            groups[name] = {"start": start, "end": end}
        if not shaped_ok or not groups:
            continue

        ordered = sorted(groups.items(), key=lambda kv: kv[1]["start"])  # check B
        cursor, tiled = 0, True
        for name, rng in ordered:
            if rng["start"] != cursor:
                failures.append(f"modality.json['{modality_type}']: '{name}' starts at {rng['start']} "
                                f"but the previous group ended at {cursor}")
                tiled = False
                break
            cursor = rng["end"]
        if tiled and cursor != declared:
            failures.append(f"modality.json['{modality_type}']: groups cover [0,{cursor}) but "
                            f"info.json features['{feature_key}'] declares {declared} dims")
            tiled = False
        if tiled:
            validated[modality_type] = groups

    if "state" in validated and "action" in validated:  # check D
        s, a = set(validated["state"]), set(validated["action"])
        if s != a:
            failures.append(f"state groups {sorted(s)} != action groups {sorted(a)}")
    return validated


def _check_video_keys(modality: dict, features: dict, failures: list[str]) -> dict[str, str]:
    """Check E. Returns {view: original_key}."""
    section = modality.get("video")
    if not isinstance(section, dict) or not section:
        failures.append("modality.json['video'] is missing or empty")
        return {}
    resolved = {}
    for view, spec in section.items():
        original_key = (spec or {}).get("original_key", f"observation.images.{view}")
        if original_key in features:
            resolved[view] = original_key
        else:
            failures.append(f"video view '{view}' resolves to '{original_key}', not in info.json['features']")
    return resolved


def _episode_paths(info: dict, episode_index: int, video_keys: dict[str, str]) -> list[str]:
    chunk_idx = episode_index // int(info["chunks_size"])
    paths = [info["data_path"].format(episode_chunk=chunk_idx, episode_index=episode_index)]
    pattern = info.get("video_path")
    if pattern:
        for original_key in video_keys.values():
            paths.append(pattern.format(episode_chunk=chunk_idx, video_key=original_key,
                                        episode_index=episode_index))
    return paths


def validate_dataset(meta_texts: dict[str, str | None], present_keys: Iterable[str]) -> dict:
    """Validate a staged LeRobot/GR00T dataset.

    ``meta_texts`` maps each of META_FILES to its text (None when absent); ``present_keys`` is every
    dataset-relative key that was staged (``meta/info.json``, ``data/chunk-000/episode_000000.parquet``,
    ``videos/...``). Raises DatasetInvalid listing every failure; returns a small summary otherwise.
    """
    failures: list[str] = []
    present = set(k.lstrip("/") for k in present_keys)

    for name in META_FILES:  # check A
        if meta_texts.get(name) is None:
            failures.append(f"{name} is missing")
    if failures:
        raise DatasetInvalid(failures)

    info = parse_json(meta_texts["meta/info.json"], "meta/info.json", failures)
    modality = parse_json(meta_texts["meta/modality.json"], "meta/modality.json", failures)
    episodes = parse_jsonl(meta_texts["meta/episodes.jsonl"], "meta/episodes.jsonl", failures)
    tasks = parse_jsonl(meta_texts["meta/tasks.jsonl"], "meta/tasks.jsonl", failures)
    if failures or not isinstance(info, dict) or not isinstance(modality, dict):
        raise DatasetInvalid(failures or ["meta/info.json and meta/modality.json must be objects"])
    for key in ("data_path", "chunks_size"):
        if key not in info:
            failures.append(f"info.json is missing required key '{key}'")
    if failures:
        raise DatasetInvalid(failures)

    features = info.get("features", {}) or {}
    _check_groups(modality, features, failures)
    video_keys = _check_video_keys(modality, features, failures)

    tasks_map = {t["task_index"]: t.get("task") for t in tasks if isinstance(t, dict) and "task_index" in t}
    if not tasks_map:
        failures.append("tasks.jsonl yields no task index")

    resolved, missing, no_length = 0, 0, 0  # check G
    for ep in episodes:
        if not isinstance(ep, dict) or ep.get("episode_index") is None:
            failures.append("episodes.jsonl entry has no 'episode_index'")
            missing += 1
            continue
        if "length" not in ep:
            no_length += 1
            continue
        absent = [p for p in _episode_paths(info, int(ep["episode_index"]), video_keys) if p not in present]
        if absent:
            missing += 1
            if missing <= MAX_REPORTED_MISSING_EPISODES:
                failures.append(f"episode {ep['episode_index']}: missing {', '.join(absent)}")
            continue
        resolved += 1
    if no_length:
        failures.append(f"{no_length} episode(s) have no 'length' field")
    if missing > MAX_REPORTED_MISSING_EPISODES:
        failures.append(f"... and {missing - MAX_REPORTED_MISSING_EPISODES} further episode(s) with missing files")
    if resolved == 0 and not failures:
        failures.append("no episode resolved to an existing parquet file")

    if failures:
        raise DatasetInvalid(failures)
    return {"episodes": resolved, "tasks": len(tasks_map), "videoKeys": sorted(video_keys)}


def check_batch_divides_gpus(batch_size: int, gpus: int | None) -> None:
    """Refuse a GLOBAL batch the trainer cannot split across the job definition's GPUs.
    ``gpus`` None means the count could not be read; the upstream script treats that as advisory
    and lets the submission through, so this does too."""
    if not gpus:
        return
    if batch_size % gpus != 0 or batch_size < gpus:
        suggestion = gpus * 2
        raise ValueError(
            f"batchSize {batch_size} is not usable on a job definition requesting {gpus} GPU(s): the "
            f"trainer asserts global_batch_size % num_gpus == 0 before dividing it into the per-device "
            f"batch, so the job would fail at startup. Use batchSize {suggestion} (grad-accum "
            f"{max(1, 16 // suggestion)} keeps the effective batch at 16) or a {batch_size}-GPU-or-fewer "
            f"job definition.")
