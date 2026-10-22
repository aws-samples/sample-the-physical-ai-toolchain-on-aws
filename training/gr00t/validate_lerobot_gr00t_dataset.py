#!/usr/bin/env python3
"""
Validate that a LeRobot dataset will actually load in the GR00T N1.7 container.

This re-implements, outside the container, every assertion
`gr00t/data/dataset/lerobot_episode_loader.py` makes while loading a dataset, so
a malformed dataset fails here in seconds instead of 20 minutes into a GPU job
that has already paid for an instance and a 22.6 GB image pull.

Checks (each printed as PASS / FAIL / WARN):
  A  meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl and meta/modality.json
     all exist and parse.
  B  modality.json state/action groups have integer start/end, do not overlap,
     and tile [0, info.features[...].shape[0]) exactly.
  C  no state/action group carries an `original_key`. A present original_key is
     used verbatim as the stats.json lookup key by get_dataset_statistics(), and
     generate_stats() only ever writes feature keys (observation.state, action,
     timestamp) — never a group name — so it would KeyError at startup.
  D  the state and action group NAME SETS are identical. A RELATIVE action
     resolves its reference state as
     `state_key = action_config.state_key or <action group name>` followed by
     `assert state_key in state` (state_action_processor.py:377-386).
  E  every video entry's original_key appears verbatim in info.json["features"]
     (the loader asserts `original_key in self.feature_config`).
  F  every annotation entry's original_key is a real column in the sampled
     parquet files (the loader reads it as a column and maps it through
     tasks.jsonl).
  G  for every episode in episodes.jsonl: `length` is present, the parquet path
     formatted from info.json["data_path"] exists, and the mp4 path formatted
     from info.json["video_path"] exists for every declared video key.
  H  sampled parquet observation.state / action list widths equal the declared
     shapes, and the row count is >= the action horizon (the loader slices
     `df.iloc[:min(len(df), length)]` and the processor needs
     max(action delta_indices)+1 future steps).
  I  every task_index in the sampled parquets resolves in tasks.jsonl
     (`self.tasks_map[x]` is an unguarded dict lookup).
  J  WARN only: action dimensions whose global range is zero. These are safe —
     state_action_processor.py:169 clamps the normalization denominator with
     `np.maximum(range_vals, 1e-8)` — but a constant dimension carries no signal
     and is worth knowing about before reading a loss curve.

`meta/stats.json` is intentionally NOT required: it is generated on rank 0
inside the container by gr00t/data/dataset/factory.py before the loader asserts
on it, and train_entrypoint_n17.py deletes any shipped copy.

Usage (from the repo root):
    python training/gr00t/validate_lerobot_gr00t_dataset.py \
        --dataset-dir /tmp/g1scratch/pick_place_data --sample-episodes 201

Exit code is 0 only when every check passes. Warnings do not affect it.

Prerequisites:
    pip install pyarrow     # parquet reads (already a `pai` CLI dep)
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Loader defaults, from gr00t/data/dataset/lerobot_episode_loader.py.
DEFAULT_COLUMN_NAMES = {"state": "observation.state", "action": "action"}
FEATURE_FOR_MODALITY = DEFAULT_COLUMN_NAMES

# Matches `delta_indices=list(range(0, 16))` in
# containers/gr00t-training/g1_dex3_modality_config.py (and the UR3 config).
# Override with --action-horizon if a config uses a different horizon.
DEFAULT_ACTION_HORIZON = 16


class Report:
    """Collects pass/fail/warn lines and decides the exit code."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, msg: str) -> None:
        print(f"  [PASS] {msg}")

    def fail(self, msg: str) -> None:
        print(f"  [FAIL] {msg}")
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        print(f"  [WARN] {msg}")
        self.warnings.append(msg)

    def check(self, condition: bool, msg: str) -> bool:
        if condition:
            self.ok(msg)
        else:
            self.fail(msg)
        return bool(condition)


def load_json(path: Path, report: Report) -> dict | None:
    if not path.exists():
        report.fail(f"missing {path}")
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        report.fail(f"{path} does not parse: {exc}")
        return None
    report.ok(f"{path.name} exists and parses")
    return data


def load_jsonl(path: Path, report: Report) -> list[dict] | None:
    if not path.exists():
        report.fail(f"missing {path}")
        return None
    rows = []
    try:
        with open(path) as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    report.fail(f"{path}:{lineno} does not parse: {exc}")
                    return None
    except OSError as exc:
        report.fail(f"cannot read {path}: {exc}")
        return None
    report.ok(f"{path.name} exists and parses ({len(rows)} line(s))")
    return rows


def check_groups(
    modality: dict,
    features: dict,
    report: Report,
) -> dict[str, dict[str, dict[str, int]]]:
    """Checks B, C and D. Returns the validated {modality: {group: range}} map."""
    validated: dict[str, dict[str, dict[str, int]]] = {}

    for modality_type in ("state", "action"):
        section = modality.get(modality_type)
        if not isinstance(section, dict) or not section:
            report.fail(f"modality.json['{modality_type}'] is missing or empty")
            continue

        feature_key = FEATURE_FOR_MODALITY[modality_type]
        feature = features.get(feature_key)
        if not isinstance(feature, dict) or not feature.get("shape"):
            report.fail(f"info.json features['{feature_key}'] is missing or has no shape")
            continue
        declared = int(feature["shape"][0])

        groups: dict[str, dict[str, int]] = {}
        shaped_ok = True
        for name, spec in section.items():
            if not isinstance(spec, dict):
                report.fail(f"modality.json['{modality_type}']['{name}'] is not an object")
                shaped_ok = False
                continue
            start, end = spec.get("start"), spec.get("end")
            if not isinstance(start, int) or isinstance(start, bool) \
                    or not isinstance(end, int) or isinstance(end, bool):
                report.fail(
                    f"modality.json['{modality_type}']['{name}'] start/end must be "
                    f"integers, got {start!r}/{end!r}"
                )
                shaped_ok = False
                continue
            if end <= start:
                report.fail(
                    f"modality.json['{modality_type}']['{name}'] has end <= start "
                    f"({start}:{end}) — the slice would be empty"
                )
                shaped_ok = False
                continue
            # Check C: an original_key here redirects the stats.json lookup.
            if "original_key" in spec:
                report.fail(
                    f"modality.json['{modality_type}']['{name}'] sets "
                    f"original_key={spec['original_key']!r}; state/action groups "
                    f"must omit it so get_dataset_statistics() falls back to "
                    f"'{feature_key}', which is what generate_stats writes"
                )
                shaped_ok = False
            groups[name] = {"start": start, "end": end}

        if not shaped_ok or not groups:
            continue
        report.ok(f"modality.json['{modality_type}']: {len(groups)} group(s) with integer ranges")
        report.ok(f"modality.json['{modality_type}']: no group sets original_key")

        # Check B: non-overlapping and tiling [0, declared).
        ordered = sorted(groups.items(), key=lambda kv: kv[1]["start"])
        cursor = 0
        tiled = True
        for name, rng in ordered:
            if rng["start"] != cursor:
                report.fail(
                    f"modality.json['{modality_type}']: '{name}' starts at "
                    f"{rng['start']} but the previous group ended at {cursor} "
                    f"({'overlap' if rng['start'] < cursor else 'gap'})"
                )
                tiled = False
                break
            cursor = rng["end"]
        if tiled and cursor != declared:
            report.fail(
                f"modality.json['{modality_type}']: groups cover [0,{cursor}) but "
                f"info.json features['{feature_key}'] declares {declared} dims"
            )
            tiled = False
        if tiled:
            layout = ", ".join(f"{n} [{r['start']}:{r['end']})" for n, r in ordered)
            report.ok(
                f"modality.json['{modality_type}']: groups tile [0,{declared}) exactly "
                f"— {layout}"
            )
            validated[modality_type] = groups

    # Check D: identical group name sets.
    if "state" in validated and "action" in validated:
        state_names, action_names = set(validated["state"]), set(validated["action"])
        report.check(
            state_names == action_names,
            f"state and action group names match ({sorted(state_names)})"
            if state_names == action_names
            else f"state groups {sorted(state_names)} != action groups {sorted(action_names)} "
                 f"— a RELATIVE action asserts its group name exists in state",
        )
    return validated


def check_video_keys(modality: dict, features: dict, report: Report) -> dict[str, str]:
    """Check E. Returns {view: original_key}."""
    section = modality.get("video")
    if not isinstance(section, dict) or not section:
        report.fail("modality.json['video'] is missing or empty")
        return {}
    resolved = {}
    for view, spec in section.items():
        original_key = (spec or {}).get("original_key", f"observation.images.{view}")
        if original_key in features:
            report.ok(f"video view '{view}' -> features['{original_key}']")
            resolved[view] = original_key
        else:
            report.fail(
                f"video view '{view}' resolves to '{original_key}', which is not in "
                f"info.json['features'] (the loader asserts on this)"
            )
    return resolved


def annotation_columns(modality: dict, report: Report) -> dict[str, str]:
    """Resolve {annotation subkey: parquet column}. Column existence is check F."""
    section = modality.get("annotation")
    if not isinstance(section, dict) or not section:
        report.warn(
            "modality.json has no 'annotation' section — a language modality key "
            "'annotation.X' would assert on it"
        )
        return {}
    resolved = {}
    for subkey, spec in section.items():
        resolved[subkey] = (spec or {}).get("original_key", f"annotation.{subkey}")
    return resolved


def episode_paths(
    info: dict, episode_index: int, video_keys: dict[str, str]
) -> tuple[str, list[str]]:
    """Format the parquet and mp4 relative paths for one episode index.

    Mirrors the loader: `chunk_idx = episode_index // info['chunks_size']`, and
    the video path is formatted with the ORIGINAL feature key as `video_key`.
    """
    chunk_idx = episode_index // int(info["chunks_size"])
    parquet = info["data_path"].format(episode_chunk=chunk_idx, episode_index=episode_index)
    videos = []
    pattern = info.get("video_path")
    if pattern:
        for original_key in video_keys.values():
            videos.append(
                pattern.format(
                    episode_chunk=chunk_idx,
                    video_key=original_key,
                    episode_index=episode_index,
                )
            )
    return parquet, videos


def pick_sample(episodes: list[dict], n: int) -> list[dict]:
    """Evenly spaced sample of n episodes (all of them when n >= len)."""
    if n >= len(episodes):
        return list(episodes)
    if n <= 0:
        return []
    step = len(episodes) / n
    return [episodes[min(len(episodes) - 1, int(i * step))] for i in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a LeRobot dataset against every assertion the GR00T "
                    "N1.7 episode loader makes.",
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="LeRobot dataset root (the directory containing meta/, data/, videos/)",
    )
    parser.add_argument(
        "--sample-episodes", type=int, default=5,
        help="How many episodes to open and read (evenly spaced). Path existence "
             "is checked for ALL episodes regardless. Default: 5",
    )
    parser.add_argument(
        "--action-horizon", type=int, default=DEFAULT_ACTION_HORIZON,
        help=f"max(action delta_indices)+1 from the GR00T modality config; episodes "
             f"shorter than this cannot yield a full action chunk. "
             f"Default: {DEFAULT_ACTION_HORIZON}",
    )
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "ERROR: pyarrow is required to read the parquet files.\n"
            "  Install it in this environment:  pip install pyarrow",
            file=sys.stderr,
        )
        return 2

    dataset_dir = Path(args.dataset_dir).resolve()
    meta_dir = dataset_dir / "meta"
    report = Report()

    print(f"Validating {dataset_dir}")
    print()
    print("A. metadata files")
    if not dataset_dir.is_dir():
        report.fail(f"{dataset_dir} is not a directory")
        print()
        print("FAILED (1 problem)")
        return 1

    info = load_json(meta_dir / "info.json", report)
    modality = load_json(meta_dir / "modality.json", report)
    episodes = load_jsonl(meta_dir / "episodes.jsonl", report)
    tasks = load_jsonl(meta_dir / "tasks.jsonl", report)
    if info is None or modality is None or episodes is None or tasks is None:
        print()
        print(f"FAILED ({len(report.failures)} problem(s)) — metadata is unusable, "
              f"stopping before the structural checks")
        return 1

    stats_path = meta_dir / "stats.json"
    if stats_path.exists():
        report.warn(
            "meta/stats.json is present. It is generated in-container by "
            "generate_stats() and train_entrypoint_n17.py deletes any shipped "
            "copy, so this file is dead weight (it is not an error)"
        )

    features = info.get("features", {})
    for key in ("data_path", "chunks_size"):
        if key not in info:
            report.fail(f"info.json is missing required key '{key}'")
    if report.failures:
        print()
        print(f"FAILED ({len(report.failures)} problem(s))")
        return 1

    print()
    print("B/C/D. modality.json state & action groups")
    groups = check_groups(modality, features, report)

    print()
    print("E. video keys")
    video_keys = check_video_keys(modality, features, report)

    print()
    print("F/I. annotation columns and task indices")
    annotations = annotation_columns(modality, report)
    tasks_map = {t["task_index"]: t["task"] for t in tasks if "task_index" in t}
    report.check(bool(tasks_map), f"tasks.jsonl yields {len(tasks_map)} task index/indices")

    print()
    print("G. per-episode parquet and video files")
    resolved_episodes = 0
    missing = 0
    no_length = 0
    for ep in episodes:
        ep_index = ep.get("episode_index")
        if ep_index is None:
            report.fail(f"episodes.jsonl entry has no 'episode_index': {ep}")
            missing += 1
            continue
        if "length" not in ep:
            no_length += 1
            continue
        parquet_rel, video_rels = episode_paths(info, int(ep_index), video_keys)
        paths = [parquet_rel] + video_rels
        absent = [p for p in paths if not (dataset_dir / p).exists()]
        if absent:
            missing += 1
            if missing <= 5:
                report.fail(f"episode {ep_index}: missing {', '.join(absent)}")
            continue
        resolved_episodes += 1

    if no_length:
        report.fail(
            f"{no_length} episode(s) in episodes.jsonl have no 'length' field "
            f"(get_episode_lengths() does int(ep_meta['length']))"
        )
    if missing > 5:
        report.fail(f"... and {missing - 5} further episode(s) with missing files")
    report.check(
        resolved_episodes == len(episodes),
        f"{resolved_episodes}/{len(episodes)} episodes resolved "
        f"(parquet + {len(video_keys)} video file(s) each)",
    )

    print()
    print(f"H. sampled parquet contents (--sample-episodes {args.sample_episodes})")
    sample = pick_sample(episodes, args.sample_episodes)
    action_dim = int(features.get("action", {}).get("shape", [0])[0])
    action_min = [math.inf] * action_dim
    action_max = [-math.inf] * action_dim
    width_failures = 0
    short_episodes = []
    unknown_tasks: set[int] = set()
    missing_columns: set[str] = set()
    read_failures = 0
    rows_read = 0

    for ep in sample:
        ep_index = int(ep["episode_index"])
        parquet_rel, _ = episode_paths(info, ep_index, video_keys)
        parquet_path = dataset_dir / parquet_rel
        try:
            table = pq.read_table(parquet_path)
        except Exception as exc:  # noqa: BLE001 - surface any reader error as a failure
            read_failures += 1
            if read_failures <= 5:
                report.fail(f"episode {ep_index}: cannot read {parquet_path}: {exc}")
            continue

        columns = set(table.column_names)
        for column in list(annotations.values()) + list(DEFAULT_COLUMN_NAMES.values()):
            if column not in columns:
                missing_columns.add(column)

        nrows = table.num_rows
        rows_read += nrows
        if nrows < args.action_horizon:
            short_episodes.append((ep_index, nrows))
        declared_length = int(ep["length"])
        if nrows < declared_length:
            report.warn(
                f"episode {ep_index}: parquet has {nrows} rows but episodes.jsonl "
                f"says {declared_length}; the loader truncates to the smaller value"
            )

        # Width check for state and action, plus per-dimension action range for J.
        for modality_type, column in DEFAULT_COLUMN_NAMES.items():
            if column not in columns:
                continue
            declared_dim = int(features.get(column, {}).get("shape", [0])[0])
            values = table.column(column).to_pylist()
            bad = next((len(v) for v in values if v is None or len(v) != declared_dim), None)
            if bad is not None:
                width_failures += 1
                if width_failures <= 5:
                    report.fail(
                        f"episode {ep_index}: {column} list width {bad} != declared "
                        f"shape [{declared_dim}]"
                    )
            if column == "action" and action_dim:
                for row in values:
                    if row is None or len(row) != action_dim:
                        continue
                    for i, value in enumerate(row):
                        if value < action_min[i]:
                            action_min[i] = value
                        if value > action_max[i]:
                            action_max[i] = value

        if "task_index" in columns:
            for value in set(table.column("task_index").to_pylist()):
                if value not in tasks_map:
                    unknown_tasks.add(value)

    report.check(
        read_failures == 0,
        f"{len(sample) - read_failures}/{len(sample)} sampled parquet file(s) read "
        f"({rows_read} rows)",
    )
    if missing_columns:
        report.fail(
            f"sampled parquet files are missing required column(s): "
            f"{sorted(missing_columns)}"
        )
    else:
        report.ok(
            f"sampled parquet files carry {sorted(set(list(annotations.values()) + list(DEFAULT_COLUMN_NAMES.values())))}"
        )
    report.check(
        width_failures == 0,
        f"observation.state / action list widths match the declared shapes in all "
        f"{len(sample) - read_failures} sampled episode(s)",
    )
    if short_episodes:
        report.fail(
            f"{len(short_episodes)} sampled episode(s) have fewer rows than the "
            f"action horizon {args.action_horizon}: "
            f"{short_episodes[:5]}{' ...' if len(short_episodes) > 5 else ''}"
        )
    else:
        report.ok(f"every sampled episode has >= {args.action_horizon} rows (action horizon)")
    report.check(
        not unknown_tasks,
        f"every sampled task_index resolves in tasks.jsonl"
        if not unknown_tasks
        else f"task_index value(s) {sorted(unknown_tasks)} are not in tasks.jsonl "
             f"(tasks_map[x] is an unguarded lookup)",
    )

    print()
    print("J. constant action dimensions (warnings only)")
    action_names = features.get("action", {}).get("names") or []
    constant = []
    for i in range(action_dim):
        if action_min[i] == math.inf:
            continue
        if action_max[i] - action_min[i] == 0.0:
            name = action_names[i] if i < len(action_names) else f"dim{i}"
            constant.append((i, name))
    if constant:
        for i, name in constant:
            report.warn(
                f"action dim {i} ({name}) is constant at {action_min[i]} across the "
                f"sample — zero range. Safe: state_action_processor.py clamps the "
                f"normalization denominator at 1e-8, but this dimension carries no "
                f"learning signal"
            )
    else:
        report.ok("no action dimension has zero range across the sample")

    print()
    if report.failures:
        print(f"FAILED — {len(report.failures)} problem(s), {len(report.warnings)} warning(s)")
        for msg in report.failures:
            print(f"  - {msg}")
        return 1
    print(f"PASSED — 0 problems, {len(report.warnings)} warning(s)")
    print(f"  {resolved_episodes}/{len(episodes)} episodes resolved, "
          f"{len(sample)} sampled, {rows_read} rows read")
    print("  meta/stats.json is generated in-container; it is not required here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
