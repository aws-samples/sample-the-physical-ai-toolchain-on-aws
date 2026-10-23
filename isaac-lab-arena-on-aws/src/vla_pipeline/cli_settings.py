"""Bind the shared toolchain configuration to ordinary Arena CLI arguments.

Explicit flags win. Saved deployment/run records remain authoritative for
reconnecting, and configuration contains references rather than token values.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def read_settings(path=None):
    try:
        from pai.config import CONFIG_PATH, load
    except ModuleNotFoundError as exc:
        if exc.name not in {"pai", "pai.config"}:
            raise
        if path is not None:
            raise ValueError("--config requires the shared toolchain package; install it from the repository root")
        return {}, None
    selected = Path(path).expanduser().resolve() if path else CONFIG_PATH
    document = load(selected)
    if not isinstance(document, dict) or not isinstance(document.get("arena", {}), dict):
        raise ValueError(f"{selected}: arena must be a JSON object")
    return document.get("arena", {}), str(selected)


def section(settings, name):
    value = settings.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config.json: arena.{name} must be an object")
    return value


def apply_settings(args):
    """Fill unspecified choices; never change a resumed operation's saved inputs."""
    if args.command == "cells":
        return
    settings, path = read_settings(args.config)
    args.config_path = path
    account = settings.get("account_id")
    args.configured_account = str(account) if account is not None else None

    def default(field, value):
        if getattr(args, field, None) is None and value is not None:
            setattr(args, field, value)

    if args.command == "deploy":
        default("name", settings.get("deployment_name"))
        default("profile", settings.get("profile"))
        default("region", settings.get("region", "us-east-1"))
        default("account_id", settings.get("account_id"))
        if not args.name:
            raise ValueError("Supply --name or set arena.deployment_name in config.json")
        if args.resume:
            return
        if not args.use_existing and not args.prepare_host:
            default("cell", settings.get("prepare_cells"))
            default("hf_token_file", settings.get("hf_token_file"))
            default("ngc_token_file", settings.get("ngc_token_file"))
            if settings:
                default("project", args.name)
            if args.create_local_host:
                default("gpu_zone", section(settings, "local").get("availability_zone"))
        if args.account_id is not None and not re.fullmatch(r"\d{12}", str(args.account_id)):
            raise ValueError("Set arena.account_id to the twelve-digit AWS account, or supply --account-id")
        if args.account_id is not None:
            args.account_id = str(args.account_id)
    elif args.command == "run":
        default("deployment", settings.get("deployment_name"))
        if not args.deployment:
            raise ValueError("Supply --deployment or set arena.deployment_name in config.json")
        execution = section(settings, args.mode)
        run = section(settings, "run")
        if not any((args.cell, args.model, args.model_version, args.simulator, args.suite)):
            default("cell", execution.get("cell"))
        default("max_runtime_seconds", run.get("max_runtime_seconds"))
        # Hardware defaults apply only to GPU steps that the request selects.
        from .workflow import selected_steps
        steps = selected_steps(
            args.through or ("RegisterModel" if args.mode == "managed" else "SuccessGate"),
            checkpoint=bool(args.checkpoint_s3),
        )
        if args.mode == "managed":
            from .launch_request import assignments
            instances = assignments(args.instance, "--instance")
            for step, key in (("FineTune", "train_instance_type"), ("SimEval", "eval_instance_type")):
                if step in steps and step not in instances and execution.get(key) is not None:
                    instances[step] = execution[key]
            args.instance = [f"{step}={value}" for step, value in instances.items()]
        default("record_video", run.get("record_video", False))
        if args.record_video != "auto" and type(args.record_video) is not bool:
            raise ValueError("arena.run.record_video must be true, false or \"auto\"")
        args.configuration = {
            "path": path, "account_id": settings.get("account_id"),
            "region": settings.get("region"), "deployment": args.deployment,
        }


def require_account(session, expected):
    """Check the actual credential provider before any provisioning write."""
    caller = session.client("sts").get_caller_identity()
    if expected is not None and caller["Account"] != str(expected):
        raise ValueError(
            f"AWS credentials select account {caller['Account']}; configuration requires {expected}. "
            "Authenticate with the target account's profile before deployment."
        )
    return caller


def configuration_receipt(request):
    """Hash resolved nonsecret choices; token material is never part of a run."""
    values = {key: request[key] for key in (
        "cell", "mode", "steps", "parameters", "checkpoint_s3", "sample", "configuration"
    ) if key in request}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "values": values}
