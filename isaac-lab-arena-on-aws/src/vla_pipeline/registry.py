"""Registry: resolve (model_family, simulator) -> adapter spec from config manifests.

Families, simulators, and pairs are YAML manifests under ``config/``;
``resolve()`` supplies the adapter identity consumed by the launchers.

Design constraints:
  * Adapter identity ONLY -- images, connector, use_groot_server, default suite,
    registry group, delivery mode. Per-run knobs (DoseSteps, EvalTrials,
    EvalSeed, EvalTaskIds, SuccessThreshold, instance types) stay CLI args on
    the launcher (rule 3c "core params stay").
  * Fail LOUD on a missing/malformed/mismatched manifest -- never a silent
    default (Standing rule: no fabricated values).
  * Pure/offline: no AWS, no network. The ECR base prefix is applied by the
    caller (it needs cfg.account_id/region), exactly as the launchers do today.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import re
from typing import NamedTuple, Any

import yaml

# repo_root/src/vla_pipeline/registry.py -> repo_root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CONFIG_DIR = os.environ.get("VLA_CONFIG_DIR") or os.path.join(_REPO_ROOT, "config")


#: ``DatasetS3Uri`` sentinel meaning "there is no S3 dataset -- resolve the
#: fine-tune dataset from the suite manifest's ``dataset:`` block". The literal
#: value is retained for backward compatibility with already-registered pipeline
#: definitions and with the openvla/molmoact2 train entries that compare against
#: it; the name is a historical misnomer (it predates Arena).
#: TODO(port): rename the VALUE, in one commit touching this constant and both
#: train entries together. See README.md#notes-and-limitations.
DATASET_FROM_SUITE_MANIFEST = "__LIBERO_DEFAULT__"


class RegistryError(RuntimeError):
    """Raised on any missing/malformed/mismatched manifest. Never swallowed."""


def named_cells() -> dict[str, dict]:
    """Named selections only; adapter identity still comes from the registry."""
    cells = _read_yaml(os.path.join(_CONFIG_DIR, "cells.yaml"))
    for name, cell in cells.items():
        if not isinstance(cell, dict) or set(cell) != {
            "model", "version", "simulator", "suite", "local_profile"
        }:
            raise RegistryError(f"Invalid named cell {name!r}")
        resolve(cell["model"], cell["simulator"])
        suite = resolve_suite(cell["suite"])
        if suite.simulator != cell["simulator"] or not suite.supported:
            raise RegistryError(f"Cell {name!r} selects an unsupported or mismatched suite")
    return cells


#: ``supported`` = registrable: the suite is on the validator's allowlist and a
#: launcher will submit it without an opt-in flag. It does NOT assert the cell has
#: been proven end-to-end, nor that every field needed for a run is filled in --
#: an Arena suite can be ``supported`` with ``arena.policy_config: null``, in which
#: case the launcher refuses to submit until one is supplied.
#: ``experimental`` = the manifest exists so the wiring is explicit, but the suite
#: is excluded from :func:`valid_suites` and is rejected at submit by the
#: launchers unless the caller opts in.
SUITE_STATUSES = ("supported", "experimental")

#: Allowed top-level keys in a suite manifest (typos are rejected, not ignored).
# I6: evaluation_protocol declares the episode horizon and action-chunk length the SUITE
# expects. Those were read from the checkpoint being evaluated, so a checkpoint could
# change the experiment while keeping the suite label.
_KNOWN_SUITE_KEYS = {"name", "simulator", "status", "description", "evaluation_protocol",
                     "canonical_task_ids", "dataset", "arena", "family_overrides"}

#: A suite id is also a filename component; keep it boring.
_SUITE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_]*")


@dataclasses.dataclass(frozen=True)
class SuiteDataset:
    """The FineTune dataset contract for a suite (``None`` when eval-only)."""
    repo_id: str
    subdir: str                 # "" = repository root
    revision: str | None
    copy_libero_modality: bool


@dataclasses.dataclass(frozen=True)
class ArenaRuntime:
    """Isaac Lab Arena closed-loop runtime contract -- the ``policy_runner`` argv.

    Every field maps 1:1 to a ``policy_runner.py`` argument, so a suite manifest
    fully determines the Arena command. ``embodiment``/``object`` are ``None``
    when the task takes no such flag; ``policy_config`` is ``None`` when the
    operator has not yet supplied an embodiment-matching closed-loop config (the
    launcher then refuses to submit -- see ``README.md#base-requirements``).
    """
    task: str                       # positional `task`
    embodiment: str | None       # --embodiment
    object: str | None           # --object
    policy_config: str | None    # --policy_config_yaml_path


@dataclasses.dataclass(frozen=True)
class ResolvedSuite:
    """Everything that must agree between FineTune and SimEval for one suite."""
    name: str
    simulator: str
    status: str
    description: str
    canonical_task_ids: tuple[int, ...]
    dataset: SuiteDataset | None
    arena: ArenaRuntime | None
    #: {family: {version: {"embodiment_tag": str|None, "modality_config": str|None}}}
    family_overrides: dict[str, dict[str, dict[str, str | None]]]
    #: C2: the protocol the suite's published numbers were produced under, e.g.
    #: {"n_action_steps": 8, "max_episode_steps": 720}. The manifests declared this and
    #: Validate compared it, but resolution DISCARDED it, so the serialized table never
    #: carried it and Validate's comparison skipped on absence for EVERY suite -- the check
    #: had never run once. None for simulators the parameters do not describe.
    evaluation_protocol: dict[str, int] | None

    @property
    def supported(self) -> bool:
        return self.status == "supported"

    def family_override(self, family: str, version: str) -> dict[str, str | None]:
        """Per-(family, version) embodiment wiring; ``{}`` when unspecified.

        An absent family or version is NOT an error -- it means "this suite has
        nothing to override, use the family's defaults.json".
        """
        return dict(self.family_overrides.get(family, {}).get(version, {}) or {})


@dataclasses.dataclass(frozen=True)
class ResolvedAdapterSpec:
    """The adapter identity for one (family, simulator) pair.

    Everything here is what the launcher used to hardcode; per-run knobs are
    intentionally absent (they remain launcher CLI args).
    """
    model_family: str
    simulator: str
    train_image_repo: str      # repo:tag (caller prefixes the ECR base)
    eval_image_repo: str       # repo:tag
    #: repo:tag of the image the eval image is built FROM, when the component
    #: does not build it (Isaac Lab Arena's Isaac Sim base). None for pairs whose
    #: eval image is self-contained. See README.md#baked-inputs-and-rebuilds.
    eval_base_image_repo: str | None
    arena_connector: str
    use_groot_server: str      # "true" / "false" (string, matches pipeline param)
    default_suite: str
    registry_group: str        # full group name, e.g. "vla-arena-gr00t"
    delivery_mode: str         # "baked" | "sourcedir"
    #: The capability the EVALUATOR IMAGE must attest to before submission, or None when the simulator
    #: declares none. Owned by the SIMULATOR manifest because it is a property of the image that
    #: simulator uses, not of the launcher submitting it. Hardcoding it per launcher produced both
    #: possible errors simultaneously: run_arena demanded LIBERO's capability from the Arena connector
    #: image, which can never carry it, while run_matrix demanded nothing at all.
    required_image_capability: str | None
    defaults_json: str         # repo-relative path to the family defaults.json
    #: Operational requirements the family manifest declares. These were hardcoded in every
    #: launcher, each with a different value, while the manifest fields sat unread: run_libero
    #: sent ml.g6e.12xlarge for both roles, run_arena sent ml.g6e.2xlarge for both, and
    #: run_matrix inlined ml.g6e.12xlarge -- so openvla evaluated on four large GPUs where its
    #: manifest asks for one (ml.g5.2xlarge), and Arena trained GR00T on a single GPU where the
    #: manifest asks for four. Resolving them here means a declared figure governs the run.
    train_instance: str
    eval_instance: str
    #: Per ROLE, because they run on different instances with different storage. One shared figure
    #: could not express what openvla already declared (300 train / 200 eval) and cannot be checked
    #: against two different instances' limits.
    train_volume_gb: int
    eval_volume_gb: int


def _read_yaml(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise RegistryError(f"manifest not found: {path}")
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise RegistryError(f"manifest {path} must be a YAML mapping, got "
                            f"{type(data).__name__}")
    return data


def _require(d: dict[str, Any], key: str, path: str) -> Any:
    if key not in d or d[key] is None:
        raise RegistryError(f"manifest {path} missing required field '{key}'")
    v = d[key]
    if isinstance(v, str) and not v.strip():  # empty/whitespace is a manifest typo
        raise RegistryError(f"manifest {path} required field '{key}' is empty")
    return v


def _require_instance_type(d: dict[str, Any], key: str, path: str) -> str:
    """A declared SageMaker instance type, validated in shape.

    Fails closed rather than defaulting: a launcher substituting its own instance type is how
    training reached the forward pass on half the declared GPU memory and died on
    CUDNN_STATUS_ALLOC_FAILED after the job was billed for startup.
    """
    v = _require(d, key, path)
    if not isinstance(v, str) or not v.startswith("ml."):
        raise RegistryError(f"manifest {path} field {key!r} must be a SageMaker instance type "
                            f"beginning 'ml.', got {v!r}")
    return v


def _require_volume_gb(d: dict[str, Any], key: str, path: str) -> int:
    v = _require(d, key, path)
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise RegistryError(f"manifest {path} field {key!r} must be a positive integer, got {v!r}")
    return v


def select_instance(spec: "ResolvedAdapterSpec", role: str, override: str | None,
                    acknowledged: bool = False) -> str:
    """The instance this run will use, refusing an unacknowledged departure from the declaration.

    Changing the DEFAULT does not close the finding this exists for. A real run passed
    --train-instance ml.g5.12xlarge (4x22.9 GB) for a family declaring ml.g6e.12xlarge (4x45.8 GB);
    the override won, the job was billed, and it died on CUDNN_STATUS_ALLOC_FAILED in the DDP forward
    pass. An override that silently wins over a declared requirement is the defect.

    Deliberately NOT an approved-instance list per family. Both reviewers rejected that: a list is a
    second declaration of the same fact and would drift from the first, which is the shape of every
    finding this work has closed. The declared value is the approved value; a departure is allowed but
    must be stated, so it appears in the command and in the log rather than passing silently.
    """
    if role not in ("train", "eval"):
        raise RegistryError(f"unknown instance role {role!r}")
    declared = spec.train_instance if role == "train" else spec.eval_instance
    if override is None:
        return declared
    if not isinstance(override, str) or not override.startswith("ml."):
        raise RegistryError(f"{role} instance override must be a SageMaker instance type beginning "
                            f"'ml.', got {override!r}")
    if override != declared and not acknowledged:
        raise RegistryError(
            f"{spec.model_family} declares {declared!r} for {role}, and {override!r} was requested. "
            f"Substituting hardware silently is how a run reached the training forward pass on half "
            f"the declared GPU memory and died after being billed. Pass "
            f"--acknowledge-instance-override to state the departure deliberately.")
    return override


def select_volume_gb(spec: "ResolvedAdapterSpec", role: str, override: int | None) -> int:
    """The volume for this role. Absence is `is None`, so an explicit 0 is an error, not a default."""
    if role not in ("train", "eval"):
        raise RegistryError(f"unknown volume role {role!r}")
    if override is None:
        return spec.train_volume_gb if role == "train" else spec.eval_volume_gb
    if isinstance(override, bool) or not isinstance(override, int) or override <= 0:
        raise RegistryError(f"{role} volume override must be a positive integer, got {override!r}")
    return override


def _log_unverified(instance_type: str, volume_gb: int, why: str) -> None:
    """Say plainly that the check did NOT run. Silence here would read as a pass."""
    import sys
    print(f"[registry] UNVERIFIED: could not confirm {instance_type} accepts a {volume_gb} GB volume "
          f"({why}). CreateTrainingJob still enforces the real limit; a local-NVMe instance rejects a "
          f"volume larger than its local total.", file=sys.stderr, flush=True)


def check_storage_fits(instance_type: str, volume_gb: int, ec2_client=None, *,
                       require_gpu: bool = False) -> None:
    """Refuse a volume larger than the instance's LOCAL instance storage.

    Instances with local NVMe cap VolumeSizeInGB at the local total: a real submission was rejected at
    CreateTrainingJob with

        Invalid VolumeSizeInGB: 300 GB. The requested instance type ml.g6e.xlarge includes local
        instance storage with a fixed total size of 250 GB.

    It is a CAP, not a prohibition -- ml.g5.12xlarge and ml.g6e.12xlarge both have 3800 GB and accept
    300 fine. So the rule cannot be "omit the volume for local-storage types", which is what I first
    concluded and would have been wrong.

    Checked against the EC2 API rather than a hardcoded table: a table would be a second declaration
    of a fact AWS already publishes, and it would silently rot as instance families are added. An
    instance whose capabilities cannot be read is NOT assumed compatible -- it is reported, because
    guessing here is what produced a rejected job.

    ec2_client is injectable so tests need no credentials.
    """
    if not isinstance(volume_gb, int) or isinstance(volume_gb, bool) or volume_gb <= 0:
        raise RegistryError(f"volume must be a positive integer, got {volume_gb!r}")
    if not (isinstance(instance_type, str) and instance_type.startswith("ml.")):
        raise RegistryError(f"instance type must begin 'ml.', got {instance_type!r}")

    if ec2_client is None:
        try:
            import boto3
            ec2_client = boto3.client("ec2")
        except Exception:                              # pragma: no cover - env dependent
            ec2_client = None

    # The EC2 API uses the bare type, without SageMaker's "ml." prefix.
    bare = instance_type[len("ml."):]
    if ec2_client is None:
        _log_unverified(instance_type, volume_gb, "no EC2 client available")
        return
    try:
        info = ec2_client.describe_instance_types(InstanceTypes=[bare])["InstanceTypes"]
    except Exception as exc:
        # UNVERIFIED, not verified-compatible. This runs in unit tests with no credentials and on
        # machines whose role lacks ec2:DescribeInstanceTypes, and hard-failing there would block
        # every submission on a check that is an EARLY WARNING -- CreateTrainingJob still enforces the
        # real limit, so the cost of not knowing is a fast, free rejection rather than a wrong result.
        # Said out loud so it is never mistaken for a pass.
        _log_unverified(instance_type, volume_gb, str(exc))
        return
    if not info:
        raise RegistryError(f"EC2 does not recognise instance type {bare!r}")

    if require_gpu and not any(
        gpu.get("Count", 0) > 0 and gpu.get("Manufacturer", "").lower() == "nvidia"
        for gpu in info[0].get("GpuInfo", {}).get("Gpus", [])
    ):
        raise RegistryError(
            f"{instance_type} has no NVIDIA GPU. FineTune and SimEval require CUDA GPU instances; "
            "inspect the cells --details command and the README instance matrix.")

    storage = info[0].get("InstanceStorageInfo") or {}
    local_gb = storage.get("TotalSizeInGB")
    if local_gb is None:
        return          # EBS-only: no local-storage cap applies.
    if volume_gb > local_gb:
        raise RegistryError(
            f"{instance_type} includes local instance storage of {local_gb} GB, so a "
            f"{volume_gb} GB VolumeSizeInGB is rejected at CreateTrainingJob. Choose an instance "
            f"with more local storage, or lower the declared volume -- but do NOT lower it below what "
            f"the workload needs, which is how a run fills its disk part-way through.")


def _family_path(family: str) -> str:
    return os.path.join(_CONFIG_DIR, "families", f"{family}.yaml")


def _suite_path(suite: str) -> str:
    return os.path.join(_CONFIG_DIR, "suites", f"{suite}.yaml")


def _sim_path(sim: str) -> str:
    return os.path.join(_CONFIG_DIR, "simulators", f"{sim}.yaml")


def _pair_path(family: str, sim: str) -> str:
    return os.path.join(_CONFIG_DIR, "pairs", f"{family}--{sim}.yaml")


@functools.cache
def _load_family(family: str) -> dict[str, Any]:
    p = _family_path(family)
    m = _read_yaml(p)
    if m.get("name") != family:
        raise RegistryError(
            f"family manifest {p} declares name={m.get('name')!r} but is "
            f"loaded as {family!r}")
    _require(m, "train_image", p)
    _require(m, "defaults_json", p)
    return m


@functools.cache
def _load_simulator(sim: str) -> dict[str, Any]:
    p = _sim_path(sim)
    m = _read_yaml(p)
    if m.get("name") != sim:
        raise RegistryError(
            f"simulator manifest {p} declares name={m.get('name')!r} but is "
            f"loaded as {sim!r}")
    _require(m, "delivery_mode", p)
    return m


@functools.cache
def resolve(model_family: str, simulator: str) -> ResolvedAdapterSpec:
    """Resolve a (family, simulator) pair to its adapter spec. Fail loud."""
    pair_p = _pair_path(model_family, simulator)
    pair = _read_yaml(pair_p)
    # A pair manifest must agree with its own filename -- no silent mismatch.
    if pair.get("family") != model_family or pair.get("simulator") != simulator:
        raise RegistryError(
            f"pair manifest {pair_p} declares "
            f"family={pair.get('family')!r} simulator={pair.get('simulator')!r} "
            f"but was requested as ({model_family!r}, {simulator!r})")

    family = _load_family(model_family)
    sim = _load_simulator(simulator)

    prefix = _require(pair, "registry_group_prefix", pair_p)
    # Validate the shell-routing + delivery values (fail loud on a bad manifest:
    # an unquoted YAML `true` parses to Python True -> str "True", which the shell
    # entrypoint's `== "true"` compare would miss, silently misrouting the eval).
    ugs = str(_require(pair, "use_groot_server", pair_p)).lower()
    if ugs not in ("true", "false"):
        raise RegistryError(
            f"pair {pair_p} use_groot_server must be 'true' or 'false', got {ugs!r}")
    delivery = _require(sim, "delivery_mode", _sim_path(simulator))
    if delivery not in ("baked", "sourcedir"):
        raise RegistryError(
            f"simulator {_sim_path(simulator)} delivery_mode must be 'baked' or "
            f"'sourcedir', got {delivery!r}")
    return ResolvedAdapterSpec(
        model_family=model_family,
        simulator=simulator,
        # Optional by design: absent means nothing to verify at SUBMIT time, which is not the same as
        # unverified at runtime -- the evaluator still checks what it needs when it starts.
        required_image_capability=(sim.get("required_image_capability") or None),
        train_image_repo=_require(family, "train_image", _family_path(model_family)),
        eval_image_repo=_require(pair, "eval_image", pair_p),
        eval_base_image_repo=pair.get("eval_base_image"),
        arena_connector=_require(pair, "connector", pair_p),
        use_groot_server=ugs,
        default_suite=_require(pair, "default_suite", pair_p),
        registry_group=f"{prefix}-{model_family}",
        delivery_mode=delivery,
        defaults_json=_require(family, "defaults_json", _family_path(model_family)),
        train_instance=_require_instance_type(family, "train_instance",
                                              _family_path(model_family)),
        eval_instance=_require_instance_type(family, "eval_instance",
                                             _family_path(model_family)),
        train_volume_gb=_require_volume_gb(family, "train_volume_gb",
                                          _family_path(model_family)),
        eval_volume_gb=_require_volume_gb(family, "eval_volume_gb",
                                         _family_path(model_family)),
    )


def _scan(kind: str) -> list[str]:
    d = os.path.join(_CONFIG_DIR, kind)
    if not os.path.isdir(d):
        raise RegistryError(f"config dir not found: {d}")
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".yaml"))


def clear_caches() -> None:
    """Drop every memoized manifest read.

    The loaders are ``lru_cache``d on caller-supplied strings, so a test that
    points ``_CONFIG_DIR`` at a temp directory must reset them or it poisons
    sibling tests (and vice versa). Also useful in a long-lived process that
    rewrites manifests.
    """
    for fn in (_load_family, _load_simulator, resolve, resolve_suite):
        fn.cache_clear()


def list_models() -> list[str]:
    """All family ids that have a manifest."""
    return _scan("families")


def list_simulators() -> list[str]:
    """All simulator ids that have a manifest."""
    return _scan("simulators")


def list_supported_pairs() -> list[tuple[str, str]]:
    """All (family, simulator) pairs that have a pair manifest."""
    out: list[tuple[str, str]] = []
    for stem in _scan("pairs"):
        if "--" not in stem:
            raise RegistryError(f"pair manifest '{stem}.yaml' must be named "
                                f"'<family>--<simulator>.yaml'")
        fam, sim = stem.split("--", 1)
        out.append((fam, sim))
    return out


def load_family_defaults(model_family: str) -> dict[str, Any]:
    """Load the family's defaults.json (input_config_schema + provenance_keys).

    Single source of truth. Fail loud if the referenced file is missing/malformed.
    """
    family = _load_family(model_family)
    rel = family["defaults_json"]
    p = os.path.join(_REPO_ROOT, rel)
    if not os.path.isfile(p):
        raise RegistryError(
            f"family {model_family!r} defaults_json points at {rel!r} "
            f"but {p} does not exist")
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError) as e:
        raise RegistryError(f"family {model_family!r} defaults.json {p} is "
                            f"unreadable/malformed: {e}") from e


def family_schemas() -> dict[str, dict[str, Any]]:
    """Per-family {input_config_schema, provenance_keys} for the strict validator.

    Shape matches validator.validate_report(family_schemas=).
    """
    out: dict[str, dict[str, Any]] = {}
    for fam in list_models():
        d = load_family_defaults(fam)
        out[fam] = {
            "input_config_schema": d.get("input_config_schema", {}),
            "provenance_keys": d.get("provenance_keys", {}),
        }
    return out


def list_suites(include_experimental: bool = True) -> list[str]:
    """All suite ids that have a manifest under ``config/suites/``."""
    names = _scan("suites")
    if include_experimental:
        return names
    return [s for s in names if resolve_suite(s).supported]


def _parse_dataset(raw: Any, path: str) -> SuiteDataset | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RegistryError(
            f"suite manifest {path} 'dataset' must be a mapping or null, got "
            f"{type(raw).__name__}")
    repo_id = _require(raw, "repo_id", path)
    subdir = raw.get("subdir", "")
    if subdir is None:
        subdir = ""
    if not isinstance(subdir, str):
        raise RegistryError(f"suite manifest {path} dataset.subdir must be a string")
    revision = raw.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise RegistryError(
            f"suite manifest {path} dataset.revision must be a string or null")
    clm = raw.get("copy_libero_modality")
    if not isinstance(clm, bool):
        # Fail loud rather than default: whether NVIDIA's LIBERO modality.json
        # overwrites the dataset's own meta/modality.json changes what the model
        # trains on, so it must be stated explicitly.
        raise RegistryError(
            f"suite manifest {path} dataset.copy_libero_modality must be an "
            f"explicit true/false, got {clm!r}")
    return SuiteDataset(repo_id=repo_id, subdir=subdir, revision=revision,
                        copy_libero_modality=clm)


def _parse_arena(raw: Any, path: str) -> ArenaRuntime | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RegistryError(
            f"suite manifest {path} 'arena' must be a mapping or null, got "
            f"{type(raw).__name__}")
    unknown = set(raw) - {"task", "embodiment", "object", "policy_config"}
    if unknown:
        raise RegistryError(
            f"suite manifest {path} arena has unknown key(s) {sorted(unknown)}")
    for opt in ("embodiment", "object", "policy_config"):
        if raw.get(opt) is not None and not isinstance(raw[opt], str):
            raise RegistryError(
                f"suite manifest {path} arena.{opt} must be a string or null")
    return ArenaRuntime(
        task=_require(raw, "task", path),
        embodiment=raw.get("embodiment"),
        object=raw.get("object"),
        policy_config=raw.get("policy_config"),
    )


_PROTOCOL_KEYS = ("n_action_steps", "max_episode_steps")


def _parse_evaluation_protocol(
        raw: Any, path: str, simulator: str) -> dict[str, int] | None:
    """C2: the protocol a suite's published numbers were produced under.

    Strict, and scoped to the simulator the parameters describe. n_action_steps and
    max_episode_steps are LIBERO protocol values; Arena takes its horizon from the Arena task
    configuration, so the block is only meaningful for 'libero' -- the same treatment the
    `arena:` block gets in reverse.

    Validate compares the protocol the evaluator CONSUMED against this, and that comparison
    previously skipped whenever the field was absent. It was absent for every suite, because
    resolution discarded it, so the check had never run once. A missing or malformed protocol
    must fail here, at resolution, rather than silently disabling a downstream check.
    """
    if simulator != "libero":
        if raw is not None:
            raise RegistryError(
                f"{path}: 'evaluation_protocol' is set but the simulator is {simulator!r}; "
                f"n_action_steps and max_episode_steps are LIBERO protocol parameters, and a "
                f"block that no evaluator reads would look like an enforced constraint.")
        return None
    if raw is None:
        raise RegistryError(
            f"{path}: 'evaluation_protocol' is required for libero suites. Validate compares "
            f"the protocol the evaluator consumed against it, so a suite without one cannot "
            f"have its numbers compared against anything, and the comparison would silently "
            f"pass.")
    if not isinstance(raw, dict):
        raise RegistryError(
            f"{path}: 'evaluation_protocol' must be a mapping, got {type(raw).__name__}.")
    unknown = sorted(set(raw) - set(_PROTOCOL_KEYS))
    if unknown:
        raise RegistryError(
            f"{path}: 'evaluation_protocol' has unknown key(s) {unknown}. Expected exactly "
            f"{list(_PROTOCOL_KEYS)}; an unrecognised key would be silently ignored.")
    out: dict[str, int] = {}
    for key in _PROTOCOL_KEYS:
        if key not in raw:
            raise RegistryError(f"{path}: 'evaluation_protocol' is missing {key!r}.")
        value = raw[key]
        # bool is an int subclass; True would otherwise become a step count of 1.
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RegistryError(
                f"{path}: 'evaluation_protocol.{key}' must be a positive integer, "
                f"got {value!r}.")
        out[key] = value
    return out


def _parse_family_overrides(raw: Any, path: str) -> dict[str, dict[str, dict[str, str | None]]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RegistryError(
            f"suite manifest {path} 'family_overrides' must be a mapping or null")
    out: dict[str, dict[str, dict[str, str | None]]] = {}
    known_families = set(list_models())
    for family, versions in raw.items():
        # A typo'd family here would silently never apply, so reject it.
        if family not in known_families:
            raise RegistryError(
                f"suite manifest {path} family_overrides declares unknown family "
                f"{family!r}; known families: {sorted(known_families)}")
        if not isinstance(versions, dict):
            raise RegistryError(
                f"suite manifest {path} family_overrides.{family} must be a mapping "
                f"of version -> overrides")
        out[family] = {}
        for version, ov in versions.items():
            if ov is None:
                ov = {}
            if not isinstance(ov, dict):
                raise RegistryError(
                    f"suite manifest {path} family_overrides.{family}.{version} "
                    f"must be a mapping")
            unknown = set(ov) - {"embodiment_tag", "modality_config"}
            if unknown:
                raise RegistryError(
                    f"suite manifest {path} family_overrides.{family}.{version} "
                    f"has unknown key(s) {sorted(unknown)}")
            for k, v in ov.items():
                if v is not None and not isinstance(v, str):
                    raise RegistryError(
                        f"suite manifest {path} family_overrides.{family}."
                        f"{version}.{k} must be a string or null")
            out[family][str(version)] = dict(ov)
    return out


@functools.cache
def resolve_suite(suite: str) -> ResolvedSuite:
    """Resolve a suite id to its full definition. Fail loud on any defect.

    A suite manifest is the SINGLE source of truth for what a suite means: its
    simulator, canonical task ids, fine-tune dataset, per-family embodiment
    wiring, and (for Isaac Lab Arena) the closed-loop runtime contract.
    """
    # A suite id becomes a path component, and --suite is user CLI input. Restrict
    # the charset so `resolve_suite("../pairs/gr00t--isaac_arena")` is rejected
    # HERE rather than incidentally, by the name cross-check further down.
    if not _SUITE_ID_RE.fullmatch(suite):
        raise RegistryError(
            f"invalid suite id {suite!r}: must match {_SUITE_ID_RE.pattern} "
            f"(lowercase alphanumerics and underscores)")
    p = _suite_path(suite)
    m = _read_yaml(p)
    # Reject unknown TOP-LEVEL keys too, matching _parse_arena /
    # _parse_family_overrides / the eval entry's blob check: a typo'd `datasets:`
    # would otherwise silently yield an eval-only suite and fail much later, at
    # FineTune.
    unknown = set(m) - _KNOWN_SUITE_KEYS
    if unknown:
        raise RegistryError(
            f"suite manifest {p} has unknown key(s) {sorted(unknown)}; "
            f"expected {sorted(_KNOWN_SUITE_KEYS)}")
    if m.get("name") != suite:
        raise RegistryError(
            f"suite manifest {p} declares name={m.get('name')!r} but is loaded "
            f"as {suite!r}")

    simulator = _require(m, "simulator", p)
    # Cross-manifest integrity: a suite pointing at a simulator with no manifest
    # would resolve fine here and then fail deep inside a job.
    _load_simulator(simulator)

    status = m.get("status", "supported")
    if status not in SUITE_STATUSES:
        raise RegistryError(
            f"suite manifest {p} status must be one of {list(SUITE_STATUSES)}, "
            f"got {status!r}")

    ids = _require(m, "canonical_task_ids", p)
    if (not isinstance(ids, list) or not ids
            or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids)
            or sorted(set(ids)) != ids):
        raise RegistryError(
            f"suite manifest {p} canonical_task_ids must be a non-empty, sorted, "
            f"duplicate-free list of ints, got {ids!r}")

    arena = _parse_arena(m.get("arena"), p)
    # The Arena runtime contract is what makes an isaac_arena run well-formed;
    # a suite on that simulator without it cannot produce a policy_runner argv.
    if simulator == "isaac_arena" and arena is None:
        raise RegistryError(
            f"suite manifest {p} declares simulator 'isaac_arena' but has no "
            f"'arena' block -- the closed-loop runtime contract (task, "
            f"embodiment, policy_config) is required.")
    if simulator != "isaac_arena" and arena is not None:
        raise RegistryError(
            f"suite manifest {p} has an 'arena' block but simulator is "
            f"{simulator!r}; the block is only meaningful for 'isaac_arena'.")

    return ResolvedSuite(
        name=suite,
        simulator=simulator,
        status=status,
        description=str(m.get("description", "")).strip(),
        canonical_task_ids=tuple(ids),
        dataset=_parse_dataset(m.get("dataset"), p),
        arena=arena,
        family_overrides=_parse_family_overrides(m.get("family_overrides"), p),
        evaluation_protocol=_parse_evaluation_protocol(
            m.get("evaluation_protocol"), p, simulator),
    )


def suites_for_simulator(simulator: str, include_experimental: bool = False) -> list[str]:
    """Suite ids that run on ``simulator``, for the launchers' --suite help text."""
    out = [s for s in list_suites()
           if resolve_suite(s).simulator == simulator
           and (include_experimental or resolve_suite(s).supported)]
    return sorted(out)


def valid_suites() -> list[str]:
    """Suite ids a run may register under -- the validator's suite allowlist.

    Experimental suites are excluded: their manifest exists so the wiring is
    explicit, but they were never proven end-to-end, so a report claiming one
    must not pass Validate.
    """
    return sorted(s for s in list_suites() if resolve_suite(s).supported)


def suite_canonical_task_ids() -> dict[str, list[int]]:
    """Canonical task-id list per supported suite.

    Experimental suites are excluded to match :func:`valid_suites`.
    """
    return {s: list(resolve_suite(s).canonical_task_ids) for s in valid_suites()}


def suites_json() -> dict[str, Any]:
    """JSON-serializable table of every suite -- for delivery into containers.

    ``config/`` is not present inside a SageMaker container, so the suite table is
    embedded/injected (``VLA_SUITES_JSON``) exactly the way ``family_schemas`` and
    the canonical task-id map already are. Keeping ONE serialized form means the
    validator's suite allowlist and both coherence gates (suite/task,
    suite/embodiment) read the same bytes the launcher resolved.

    NOTE: the gr00t train entry does **not** read this table -- it ships in the
    SageMaker ``source_dir`` and still owns ``SUITE_DATASETS`` / ``SUITE_META`` as
    literals, CI-pinned to these manifests by ``tests/test_suites.py``. See
    ``README.md#notes-and-limitations``.
    """
    out: dict[str, Any] = {}
    for name in list_suites():
        s = resolve_suite(name)
        out[name] = {
            "name": s.name,
            "simulator": s.simulator,
            "status": s.status,
            "canonical_task_ids": list(s.canonical_task_ids),
            "dataset": (None if s.dataset is None else dataclasses.asdict(s.dataset)),
            "arena": (None if s.arena is None else dataclasses.asdict(s.arena)),
            "family_overrides": s.family_overrides,
            "evaluation_protocol": s.evaluation_protocol,
        }
    return out


_ECR_HOST = re.compile(
    r"^(?P<account>\d{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com$")


class EcrReference(NamedTuple):
    """A parsed ECR image reference. The URI already names the registry -- use it."""
    account: str | None
    region: str | None
    repository: str
    tag: str | None
    digest: str | None

    @property
    def image_id(self) -> dict:
        return {"imageDigest": self.digest} if self.digest else {"imageTag": self.tag}


def parse_ecr_reference(image_uri: str) -> EcrReference:
    """Split an image reference into registry, repository and tag-or-digest.

    Both image helpers built their ECR client from the AMBIENT session, so a west-region image
    resolved against an east-region registry: the call could reject an image that exists, attach a
    digest from the wrong registry to the requested URI, or select different bytes that happen to sit
    at the same tag. load_config supports VLA_REGION explicitly, so an ambient/config mismatch is a
    supported configuration rather than an exotic one.

    One parser, used by both helpers, because fixing the launchers individually would leave the two
    helpers interpreting the same URI differently.
    """
    remainder, _, digest = image_uri.partition("@sha256:")
    digest = f"sha256:{digest}" if digest else None
    if digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise RegistryError(f"{image_uri!r} carries a malformed digest {digest!r}")

    # The STANDARD registry rule: the first path component is a registry host if it contains a dot or a
    # colon, or is exactly "localhost". Splitting on the strict ECR pattern instead left the whole host
    # inside the repository name whenever the pattern did not match -- ECR then matches no image, and
    # every capability read reports the image as unlabelled. Worse than not parsing at all.
    host, slash, path = remainder.partition("/")
    is_registry = slash and ("." in host or ":" in host or host == "localhost")
    if not is_registry:
        account = region = None
        path = remainder
    else:
        match = _ECR_HOST.match(host)
        # Account and region come ONLY from a host that really is an ECR registry. A non-ECR registry
        # (or an unrecognised host shape) yields None, which means "use the caller's session" -- stated
        # by returning None rather than by guessing a region.
        account = match.group("account") if match else None
        region = match.group("region") if match else None

    tag = None
    if not digest:
        repo_part, colon, tag = path.rpartition(":")
        if not colon:
            raise RegistryError(
                f"{image_uri!r} names neither a tag nor a digest, so the bytes it would run cannot be "
                f"identified")
        path = repo_part
    return EcrReference(account=account, region=region, repository=path, tag=tag, digest=digest)


def _ecr_for(reference: EcrReference, ecr_client):
    """A client for the reference's OWN region, or the injected one -- refusing a known mismatch."""
    if ecr_client is not None:
        try:
            client_region = ecr_client.meta.region_name
        except Exception:
            client_region = None
        if reference.region and client_region and client_region != reference.region:
            raise RegistryError(
                f"an ECR client for {client_region} was supplied for an image in "
                f"{reference.region}; resolving it there would describe a different registry")
        return ecr_client
    import boto3
    return boto3.client("ecr", region_name=reference.region) if reference.region \
        else boto3.client("ecr")


def resolve_image_digest(image_uri: str, ecr_client=None) -> str:
    """Turn a tag reference into a DIGEST reference, so the receipt can name the bytes that ran.

    A tag is a pointer. Passing one into an attestation records which pointer was followed, not which
    image executed -- and the two diverge silently. Three Arena executions ran against an
    isaac-lab-arena image from the previous day while their results were attributed to current source,
    because the builder meant to refresh it was submitting to a CodeBuild project whose role could not
    read the source bucket. Every receipt from those runs would have looked correct. Resolving the
    digest at SUBMIT time is what makes that divergence visible.

    Already-digest references pass through unchanged. Unlike check_storage_fits, this does NOT degrade
    when ECR is unreachable: the cost of not knowing here is an unattributable result, which is the one
    thing this component exists to prevent.
    """
    reference = parse_ecr_reference(image_uri)
    if reference.digest:
        return image_uri
    client = _ecr_for(reference, ecr_client)
    # registryId from the URI, not the caller's account: without it a same-named repository in the
    # ambient account would answer for the requested one.
    kwargs = {"repositoryName": reference.repository, "imageIds": [reference.image_id]}
    if reference.account:
        kwargs["registryId"] = reference.account
    try:
        images = client.describe_images(**kwargs)["imageDetails"]
    except Exception as exc:
        raise RegistryError(
            f"cannot resolve {image_uri} to a digest: {exc}. Refusing to submit a run whose attestation "
            f"could only name a tag: a tag can be republished, and a stale one produces an "
            f"identical-looking receipt.") from exc
    if not images:
        raise RegistryError(
            f"{reference.repository}:{reference.tag} does not exist in ECR"
            f"{f' ({reference.account} / {reference.region})' if reference.account else ''}, so nothing "
            f"can be submitted against it. Build and push the image first.")
    digest = images[0].get("imageDigest")
    if not digest or not digest.startswith("sha256:"):
        raise RegistryError(
            f"ECR returned no usable digest for {reference.repository}:{reference.tag}: {digest!r}")
    # Rebuilt from the REQUESTED reference, so the digest is attached to the registry it came from.
    return f"{image_uri.rsplit(':', 1)[0]}@{digest}"


BAKED_CAPABILITIES_LABEL = "dev.physical-ai.baked-capabilities"


def _fetch_config_blob(url: str) -> bytes:
    """Fetch an image config blob over the pre-signed URL ECR returns.

    A module-level seam, not an inline lambda, so a caller that cannot pass `opener` -- a launcher
    reached through a test that stubs only boto3 -- can still substitute it. Without this the capability
    check attempted a real DNS lookup for a fake host and failed on the network rather than on anything
    the test was about.
    """
    import urllib.request
    return urllib.request.urlopen(url, timeout=120).read()


def read_image_capabilities(image_uri: str, ecr_client=None, opener=None) -> set[str]:
    """The capabilities an image ATTESTS to, read from ECR without pulling it.

    ECR's describe_images does NOT expose labels -- it returns digests, tags and sizes only, which is
    worth stating because the obvious call looks like it should work. Labels live in the image CONFIG
    blob, reached in three steps: batch_get_image for the manifest, the manifest's config digest, then
    get_download_url_for_layer to fetch that blob. About a second, and no 30 GB pull.

    The value is single-sourced with /opt/vla/.baked_env by a shared ARG in the Dockerfile, so what a
    launcher reads before submitting and what the evaluator reads at runtime cannot drift.

    Returns the parsed capability set, or an EMPTY set when the image carries no such label. Empty is
    reported by the caller as UNKNOWN, never as incapable: images built before the label existed are
    legitimately unlabelled, and refusing them would be a false negative on evidence that says nothing.
    """
    import json
    reference = parse_ecr_reference(image_uri)
    client = _ecr_for(reference, ecr_client)
    if opener is None:
        opener = _fetch_config_blob
    registry = {"registryId": reference.account} if reference.account else {}

    try:
        resp = client.batch_get_image(
            repositoryName=reference.repository, imageIds=[reference.image_id],
            acceptedMediaTypes=["application/vnd.docker.distribution.manifest.v2+json",
                                "application/vnd.oci.image.manifest.v1+json"],
            **registry)
        images = resp.get("images") or []
        if not images:
            raise RegistryError(
                f"{reference.repository} has no image matching {reference.image_id}")
        manifest = json.loads(images[0]["imageManifest"])
        config_digest = manifest["config"]["digest"]
        url = client.get_download_url_for_layer(
            repositoryName=reference.repository, layerDigest=config_digest, **registry)["downloadUrl"]
        config = json.loads(opener(url))
    except RegistryError:
        raise
    except Exception as exc:
        raise RegistryError(
            f"cannot read the capability label from {image_uri}: {exc}. This check exists so a "
            f"missing capability is found in a second at submit time instead of after a GPU node has "
            f"been provisioned.") from exc

    labels = (config.get("config") or {}).get("Labels") or {}
    raw = labels.get(BAKED_CAPABILITIES_LABEL, "")
    return {tok for tok in raw.split(":") if tok}


def require_image_capability(image_uri: str, capability: str, ecr_client=None,
                             opener=None) -> None:
    """Refuse to submit against an image that positively lacks a required capability.

    Three outcomes, kept distinct on purpose:
      - the label names the capability            -> proceed
      - the label exists and OMITS it             -> REFUSE; the image attests to what it has, and
                                                     this is not among it
      - no label at all                           -> proceed, saying UNKNOWN out loud

    A missing label can mean an older image or a builder that does not publish capability metadata.
    It does not establish that the capability is absent. The runtime guard still checks the actual
    interpreter, so an unlabelled image defers that check until the container runs.
    """
    import sys
    caps = read_image_capabilities(image_uri, ecr_client=ecr_client, opener=opener)
    if not caps:
        print(f"[registry] UNVERIFIED: {image_uri} carries no {BAKED_CAPABILITIES_LABEL} label, so "
              f"{capability!r} cannot be confirmed before submitting. This can also occur with a "
              f"new build whose builder does not publish capability metadata. The runtime capability "
              f"check remains required; this message does not establish that the image is broken.",
              file=sys.stderr, flush=True)
        return
    if capability not in caps:
        raise RegistryError(
            f"{image_uri} does not provide {capability!r}. It attests to: {sorted(caps)}. Refusing to "
            f"submit: the evaluator would provision a GPU node and only then discover the capability "
            f"is missing. Rebuild the image so the layer that provides {capability!r} runs.")
