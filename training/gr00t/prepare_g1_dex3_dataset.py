#!/usr/bin/env python3
"""
Prepare a Unitree G1 Edu + Dex3 LeRobot dataset for GR00T N1.7 fine-tuning.

The `pick_place_data` capture is already LeRobot (`codebase_version: v2.1`,
`robot_type: Unitree_G1_Dex3`, 201 episodes, 30 fps, single camera, 28-DOF
joint-space teleop). NO FORMAT CONVERSION IS NEEDED: the N1.7 episode loader
never reads `codebase_version`, it only needs the v2 layout
(`meta/info.json`, `meta/episodes.jsonl` with a `length` field, `meta/tasks.jsonl`,
`data/chunk-*/episode_*.parquet`, `videos/chunk-*/<video_key>/episode_*.mp4`)
plus one GR00T-specific file that the capture does not ship:
`meta/modality.json`.

This script authors that one file, and only that one file. It does NOT write
`meta/stats.json` or `meta/relative_stats.json` — see the note in
`print_stats_note()`: both are generated on rank 0 inside the training container
by `gr00t.data.stats.generate_stats` / `generate_rel_stats`, and
`containers/gr00t-training/train_entrypoint_n17.py` deletes any shipped copy
before training so a stale or foreign-format file cannot poison normalization.

Group boundaries are DERIVED from the `names` arrays in `meta/info.json` (by
joint-name prefix: left/right x arm/hand) and then asserted against the expected
28-DOF G1 Dex3 layout, so a differently-shaped capture fails loudly here instead
of silently mis-slicing joints during training.

Usage (from the repo root):
    # Inspect what would be written, touch nothing
    python training/gr00t/prepare_g1_dex3_dataset.py \
        --dataset-dir /tmp/g1scratch/pick_place_data --dry-run

    # Write meta/modality.json into the local dataset copy
    python training/gr00t/prepare_g1_dex3_dataset.py \
        --dataset-dir /tmp/g1scratch/pick_place_data

    # Write, then stage to the bucket the Batch instance role can read
    python training/gr00t/prepare_g1_dex3_dataset.py \
        --dataset-dir /tmp/g1scratch/pick_place_data \
        --upload --s3-uri s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/pick_place_data/

    # Then validate before spending GPU time
    python training/gr00t/validate_lerobot_gr00t_dataset.py \
        --dataset-dir /tmp/g1scratch/pick_place_data --sample-episodes 201

Work on a LOCAL COPY. The canonical capture at
s3://physical-ai-workshop-assets-<ACCOUNT_ID>/datasets/pick_place_data/ is
read-only; pull it with `aws s3 sync` and run this against the copy.

Staging (`--upload`) is not cosmetic: the Batch instance role's inline policy
grants S3 only on `arn:aws:s3:::physical-ai-dev-*`, so the container cannot read
the workshop-assets bucket. `training/gr00t/upload_dataset.py` is deliberately
not reused — it appends a `/dataset/` suffix to the prefix, which would produce
`groot-data/pick_place_data/dataset/`.

Prerequisites:
    pip install pyarrow        # parquet column check (already a `pai` CLI dep)
    aws CLI on PATH            # only for --upload
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Expected 28-DOF Unitree G1 Edu + Dex3-1 layout. DERIVED from info.json at
# runtime and compared against this; a mismatch is a hard error, not a warning.
EXPECTED_GROUPS = {
    "left_arm": (0, 7),
    "right_arm": (7, 14),
    "left_hand": (14, 21),
    "right_hand": (21, 28),
}

# Group ordering in the written modality.json. Must match the modality_keys
# order in containers/gr00t-training/g1_dex3_modality_config.py so the
# ActionConfig list lines up with the groups it describes.
GROUP_ORDER = ["left_arm", "right_arm", "left_hand", "right_hand"]

# The parquet column that carries the language handle. This capture has no
# `annotation.*` column — only `task_index`, resolved through tasks.jsonl by the
# loader's `tasks_map`. Same pattern as upstream examples/SO100/modality.json.
ANNOTATION_KEY = "human.task_description"
ANNOTATION_ORIGINAL_KEY = "task_index"

# Shipped reference copy of the GR00T modality config, copied next to the
# dataset for provenance (which config these indices were authored for).
MODALITY_CONFIG_SRC = Path("containers/gr00t-training/g1_dex3_modality_config.py")


def _classify_joint(name: str) -> str:
    """Map a G1 joint name to its modality group name.

    Names look like `kLeftShoulderPitch`, `kRightHandThumb0`. Side comes from the
    Left/Right token, kind from the presence of `Hand` (Dex3-1 fingers) vs an arm
    joint. Anything unrecognised raises — better a loud failure than a joint
    quietly landing in the wrong group.
    """
    lowered = name.lower()
    if "left" in lowered:
        side = "left"
    elif "right" in lowered:
        side = "right"
    else:
        raise ValueError(
            f"Joint '{name}' has no Left/Right token — cannot assign it to a "
            f"modality group. This script targets the Unitree G1 Edu Dex3 "
            f"28-DOF layout; see --help."
        )
    kind = "hand" if "hand" in lowered else "arm"
    return f"{side}_{kind}"


def derive_groups(names: list[str], declared_dim: int) -> dict[str, dict[str, int]]:
    """Derive contiguous {group: {start, end}} ranges from a joint-name list.

    Asserts, in order: every joint classifies; each group's indices are
    contiguous; the groups tile [0, declared_dim) with no gap and no overlap;
    and the result equals the expected G1 Dex3 layout.
    """
    if len(names) != declared_dim:
        raise ValueError(
            f"info.json declares shape [{declared_dim}] but lists {len(names)} "
            f"joint names — the dataset metadata is inconsistent."
        )

    indices: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        indices.setdefault(_classify_joint(name), []).append(idx)

    groups: dict[str, dict[str, int]] = {}
    for group in GROUP_ORDER:
        if group not in indices:
            raise ValueError(
                f"No joints classified into group '{group}'. Derived groups: "
                f"{sorted(indices)}. Expected {GROUP_ORDER}."
            )
        idxs = indices[group]
        start, end = idxs[0], idxs[-1] + 1
        if idxs != list(range(start, end)):
            raise ValueError(
                f"Group '{group}' is not contiguous in the joint ordering: "
                f"{idxs}. modality.json can only express contiguous [start:end) "
                f"slices, so this capture cannot be sliced correctly."
            )
        groups[group] = {"start": start, "end": end}

    unexpected = sorted(set(indices) - set(GROUP_ORDER))
    if unexpected:
        raise ValueError(f"Unexpected joint groups derived from info.json: {unexpected}")

    # Tiling check: sorted ranges must start at 0, be gapless, and end at dim.
    cursor = 0
    for group, rng in sorted(groups.items(), key=lambda kv: kv[1]["start"]):
        if rng["start"] != cursor:
            raise ValueError(
                f"Group '{group}' starts at {rng['start']} but the previous group "
                f"ended at {cursor} — gap or overlap in the joint tiling."
            )
        cursor = rng["end"]
    if cursor != declared_dim:
        raise ValueError(
            f"Derived groups cover [0,{cursor}) but info.json declares "
            f"{declared_dim} dimensions."
        )

    derived = {g: (groups[g]["start"], groups[g]["end"]) for g in groups}
    if derived != EXPECTED_GROUPS:
        raise ValueError(
            f"Derived group ranges {derived} do not match the expected Unitree "
            f"G1 Edu Dex3 layout {EXPECTED_GROUPS}. This dataset is not the "
            f"28-DOF 7+7+7+7 capture this script and "
            f"containers/gr00t-training/g1_dex3_modality_config.py were written "
            f"for — fix the config and the expected layout together, do not "
            f"force the ranges."
        )
    return groups


def discover_video_key(features: dict) -> tuple[str, str]:
    """Find the single camera feature. Returns (short_name, original_key).

    The loader resolves a video view as
    `modality.json["video"][view].get("original_key", f"observation.images.{view}")`
    and then asserts that key is in info.json["features"], so the original_key
    must be the full feature key verbatim.
    """
    prefix = "observation.images."
    video_keys = [
        k for k, v in features.items()
        if k.startswith(prefix) and v.get("dtype") == "video"
    ]
    if not video_keys:
        raise ValueError(
            f"No '{prefix}*' video feature found in info.json features: "
            f"{sorted(features)}"
        )
    if len(video_keys) > 1:
        raise ValueError(
            f"Expected exactly one camera view, found {len(video_keys)}: "
            f"{video_keys}. containers/gr00t-training/g1_dex3_modality_config.py "
            f"declares a single view ('cam_high'); add the extra views to both "
            f"files together."
        )
    original_key = video_keys[0]
    return original_key[len(prefix):], original_key


def sample_parquet_columns(dataset_dir: Path, info: dict) -> tuple[Path, list[str]]:
    """Return (path, column names) of the first episode's parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit(
            "ERROR: pyarrow is required to confirm the parquet columns.\n"
            "  Install it in this environment:  pip install pyarrow"
        )

    data_path = info["data_path"].format(episode_chunk=0, episode_index=0)
    parquet_path = dataset_dir / data_path
    if not parquet_path.exists():
        raise SystemExit(f"ERROR: expected first episode parquet at {parquet_path}")
    return parquet_path, list(pq.ParquetFile(parquet_path).schema_arrow.names)


def build_modality(
    groups: dict[str, dict[str, int]],
    video_short: str,
    video_original_key: str,
) -> dict:
    """Build the meta/modality.json payload.

    NOTE on the deliberate absence of `original_key` inside the state/action
    groups: `get_dataset_statistics()` treats a present `original_key` as the key
    to look up in `stats.json`, and `generate_stats` only ever writes the feature
    keys (`observation.state`, `action`, `timestamp`) — never a group name like
    `left_arm`. Omitting it makes the lookup fall back to DEFAULT_COLUMN_NAMES
    (`observation.state` for state, `action` for action), which is what exists.
    """
    group_section = {g: dict(groups[g]) for g in GROUP_ORDER}
    return {
        # Same group names and ranges in both sections: a RELATIVE action looks
        # up its reference state by the action group's own name.
        "state": group_section,
        "action": {g: dict(group_section[g]) for g in GROUP_ORDER},
        "video": {video_short: {"original_key": video_original_key}},
        "annotation": {ANNOTATION_KEY: {"original_key": ANNOTATION_ORIGINAL_KEY}},
    }


def print_stats_note() -> None:
    print()
    print("meta/stats.json and meta/relative_stats.json were NOT written, on purpose:")
    print("  * gr00t/data/dataset/factory.py calls generate_stats(dataset_path) and")
    print("    generate_rel_stats(dataset_path, embodiment_tag) on rank 0 inside the")
    print("    container, before the loader asserts stats.json exists.")
    print("  * containers/gr00t-training/train_entrypoint_n17.py deletes any shipped")
    print("    meta/stats.json, meta/relative_stats.json and meta/*.stats.json first,")
    print("    so anything staged here is discarded.")
    print("  * meta/episodes_stats.jsonl (present in this capture) is ignored by GR00T")
    print("    entirely and lacks the q01/q99 percentiles GR00T normalizes with.")


def copy_modality_config(dataset_dir: Path, dry_run: bool) -> None:
    """Copy the GR00T modality config next to the dataset for provenance."""
    src = Path(__file__).resolve().parents[2] / MODALITY_CONFIG_SRC
    if not src.exists():
        print(f"  WARNING: {MODALITY_CONFIG_SRC} not found at {src}; skipping provenance copy.")
        return
    dest = dataset_dir / "meta" / src.name
    if dry_run:
        print(f"  [dry-run] would copy {MODALITY_CONFIG_SRC} -> {dest}")
        return
    shutil.copyfile(src, dest)
    print(f"  Copied {MODALITY_CONFIG_SRC} -> {dest} (provenance)")


def upload(dataset_dir: Path, s3_uri: str, region: str, dry_run: bool) -> None:
    """Stage the prepared copy with `aws s3 sync`.

    The destination must be a bucket the Batch instance role can read
    (`arn:aws:s3:::physical-ai-dev-*`). Do not widen that policy to reach the
    workshop-assets bucket, and never sync back onto the read-only source.
    """
    if not s3_uri.startswith("s3://"):
        raise SystemExit(f"ERROR: --s3-uri must start with s3:// (got {s3_uri!r})")
    if "workshop-assets" in s3_uri:
        raise SystemExit(
            f"ERROR: refusing to write to {s3_uri} — the workshop-assets bucket "
            f"holds the canonical read-only capture. Stage to a "
            f"physical-ai-dev-datasets-<ACCOUNT_ID> prefix instead."
        )
    dest = s3_uri if s3_uri.endswith("/") else s3_uri + "/"
    cmd = [
        "aws", "s3", "sync",
        str(dataset_dir).rstrip("/") + "/",
        dest,
        "--region", region,
    ]
    print()
    print("Staging to S3:")
    print("  " + " ".join(cmd))
    if dry_run:
        print("  [dry-run] not executed")
        return
    subprocess.run(cmd, check=True)
    print(f"  Synced {dataset_dir} -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Author meta/modality.json for a Unitree G1 Edu Dex3 "
                    "LeRobot dataset so GR00T N1.7 can load it.",
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Local LeRobot dataset root (the directory containing meta/, data/, videos/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the modality.json that would be written and change nothing",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="After writing, `aws s3 sync` the prepared copy to --s3-uri",
    )
    parser.add_argument(
        "--s3-uri", default=None,
        help="Destination prefix for --upload, e.g. "
             "s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/pick_place_data/",
    )
    parser.add_argument(
        "--region", default="us-east-2",
        help="Region passed to `aws s3 sync` (default: us-east-2)",
    )
    args = parser.parse_args()

    if args.upload and not args.s3_uri:
        parser.error("--upload requires --s3-uri")

    dataset_dir = Path(args.dataset_dir).resolve()
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        print(f"ERROR: {info_path} not found — is --dataset-dir a LeRobot root?", file=sys.stderr)
        return 1

    with open(info_path) as f:
        info = json.load(f)

    print(f"Dataset:  {dataset_dir}")
    print(f"  robot_type        {info.get('robot_type')}")
    print(f"  codebase_version  {info.get('codebase_version')} "
          f"(no conversion needed: the N1.7 loader never reads this field)")
    print(f"  episodes/frames   {info.get('total_episodes')} / {info.get('total_frames')}")
    print(f"  fps               {info.get('fps')}")

    features = info.get("features", {})
    for key in ("observation.state", "action"):
        if key not in features:
            print(f"ERROR: info.json features is missing '{key}'", file=sys.stderr)
            return 1

    state_names = features["observation.state"].get("names") or []
    action_names = features["action"].get("names") or []
    if state_names != action_names:
        print(
            "ERROR: observation.state and action joint-name lists differ. The "
            "modality groups must be identical in both sections (a RELATIVE "
            "action resolves its reference state by group name), so differing "
            "orderings cannot be expressed.",
            file=sys.stderr,
        )
        return 1

    state_dim = int(features["observation.state"]["shape"][0])
    action_dim = int(features["action"]["shape"][0])
    if state_dim != action_dim:
        print(
            f"ERROR: observation.state shape [{state_dim}] != action shape "
            f"[{action_dim}]; the same group ranges cannot apply to both.",
            file=sys.stderr,
        )
        return 1

    try:
        groups = derive_groups(state_names, state_dim)
        video_short, video_original_key = discover_video_key(features)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"  state/action dim  {state_dim}")
    print("  derived groups    " + ", ".join(
        f"{g} [{groups[g]['start']}:{groups[g]['end']})" for g in GROUP_ORDER
    ))
    print(f"  video view        {video_short} -> {video_original_key}")

    parquet_path, columns = sample_parquet_columns(dataset_dir, info)
    if ANNOTATION_ORIGINAL_KEY not in columns:
        print(
            f"ERROR: '{ANNOTATION_ORIGINAL_KEY}' is not a column in {parquet_path} "
            f"(columns: {columns}). The language modality reads "
            f"modality.json['annotation'][...]['original_key'] as a parquet "
            f"column and maps it through tasks.jsonl, so it must exist.",
            file=sys.stderr,
        )
        return 1
    print(f"  annotation        {ANNOTATION_KEY} -> {ANNOTATION_ORIGINAL_KEY} "
          f"(confirmed in {parquet_path.name})")

    modality = build_modality(groups, video_short, video_original_key)
    rendered = json.dumps(modality, indent=4)

    out_path = dataset_dir / "meta" / "modality.json"
    print()
    print(f"meta/modality.json {'(dry-run, not written)' if args.dry_run else ''}".rstrip())
    print(rendered)

    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(rendered + "\n")
        print()
        print(f"  Wrote {out_path}")

    copy_modality_config(dataset_dir, args.dry_run)
    print_stats_note()

    if args.upload:
        upload(dataset_dir, args.s3_uri, args.region, args.dry_run)

    print()
    print("Next step:")
    print(f"  python training/gr00t/validate_lerobot_gr00t_dataset.py "
          f"--dataset-dir {dataset_dir} --sample-episodes {info.get('total_episodes', 1)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
