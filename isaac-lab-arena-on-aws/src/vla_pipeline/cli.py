"""Human-facing commands for the VLA workflow."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys

from .operations import Store
from .cli_context import command_context, command_name


def parser():
    prog = command_name()
    root = argparse.ArgumentParser(
        prog=prog, description="Train a VLA model, evaluate it, validate evidence and register it."
    )
    root.add_argument("--state-dir", help="Persistent CLI records; overrides VLA_STATE_DIR "
                      "(default: ~/.local/state/vla)")
    root.add_argument("--json", action="store_true", help="Print machine-readable results")
    root.add_argument("--debug", action="store_true", help="Include a traceback when a command fails")
    root.add_argument("--config", help="Shared toolchain config.json; defaults to the repository root file")
    commands = root.add_subparsers(dest="command", required=True)
    cells = commands.add_parser("cells", help="List supported model/simulator selections")
    cells.add_argument("--cell", help="Inspect one named cell from the catalog")
    cells.add_argument("--details", action="store_true",
                       help="Show declared hardware, storage, task counts and workflow steps")

    deploy = commands.add_parser(
        "deploy", help="Prepare infrastructure/images/host, or select an existing deployment"
    )
    deploy.add_argument("--name", help="Saved deployment name; default: arena.deployment_name in config.json")
    deploy.add_argument("--use-existing", action="store_true",
                        help="Read/check existing resources; never build or change infrastructure")
    deploy.add_argument("--prepare-host", action="store_true",
                        help="Prepare a supplied host using existing resources/images; no builds or Terraform")
    deploy.add_argument("--resume", action="store_true",
                        help="Continue the saved deployment; do not resubmit successful builds")
    deploy.add_argument("--plan", action="store_true", help="Show/save preparation plans without applying them")
    deploy.add_argument("--yes", action="store_true", help="Apply displayed preparation plans noninteractively")
    deploy.add_argument("--profile", help="Provisioning AWS profile; omit on an instance-role host")
    deploy.add_argument("--region", help="Application region (default from config, otherwise us-east-1)")
    deploy.add_argument("--account-id", help="Expected AWS account; default: arena.account_id in config.json")
    deploy.add_argument("--project", help="Foundation project (default: physical-ai, or saved selection)")
    deploy.add_argument("--environment", help="Resource environment suffix (default: dev)")
    deploy.add_argument("--cell", action="append", help=f"One or more cells to prepare; see {prog} cells")
    deploy.add_argument("--image", action="append", default=[], metavar="STEP=ECR_URI",
                        help="Select a published FineTune or SimEval image; repeat per step "
                        "for one cell, using full private ECR URIs")
    deploy.add_argument("--selection", help="Import a schema-version-1 saved deployment selection")
    deploy.add_argument("--development-bucket", help="Versioned S3 bucket for local checkpoints and evidence")
    deploy.add_argument("--expected-role", help="Expected EC2 instance-role name for local execution")
    deploy.add_argument("--scratch-root", help="Working path on the host's separately mounted scratch disk")
    deploy.add_argument("--local-host", help="Existing GPU EC2 instance ID to prepare or select")
    deploy.add_argument("--create-local-host", metavar="TYPE",
                        help="Create and prepare a GPU EC2 instance of this type; omit for managed-only use")
    deploy.add_argument("--gpu-zone", help="Optional availability zone for a newly created GPU EC2 instance")
    deploy.add_argument("--hf-token-file", help="Private plaintext HF token file for creating the deployment's secret")
    deploy.add_argument("--ngc-token-file", help="Private plaintext NGC token file for creating the deployment's secret")
    deploy.add_argument("--host-region", help="Region of the EC2 host; may differ from --region")
    deploy.add_argument("--hf-secret-name", help="Existing plaintext token secret (default: vla-pipeline/hf-token)")
    deploy.add_argument("--ngc-secret-name", help="Existing plaintext token secret (default: vla-pipeline/ngc-token)")
    deploy.add_argument("--input-s3-arn", action="append",
                        help="Additional managed input bucket/object ARN; repeat as required")
    deploy.add_argument("--existing-ecr-repo", action="append", default=[],
                        choices=["vla/gr00t", "vla/openvla", "vla/molmoact2", "vla/isaac-arena"],
                        help="For a new deployment, reference this existing image repository "
                        "without taking ownership; repeat for each externally owned repository")

    run = commands.add_parser(
        "run", help="Execute a cell; the default is the complete workflow",
        description="Select a cell, execution mode and budgets, or use --sample. "
                    "The full graph runs unless an explicit input or stopping point changes it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Required choices:
  Select --cell, or --model/--simulator/--suite (plus --model-version for gr00t).
  Supply --mode. Deployment, cell, hardware and runtime may come from config.json.
  FineTune requires --train-steps; SimEval requires --eval-trials.
  SuccessGate requires --threshold (0 is a workflow sample, not a quality bar).
  --sample supplies 200 training steps, 3 trials and threshold 0; explicit flags win.
  Managed mode requires an instance choice per GPU step, via config or --instance.
  Local mode uses the prepared GPU host; omit --instance.

Examples after deployment:
  {prog} run --deployment arena --cell gr00t-n16-arena --mode local \\
    --train-steps 200 --eval-trials 3 --eval-seed 100 --threshold 0 \\
    --max-runtime-seconds 21600
  {prog} run --deployment arena --cell gr00t-n16-arena --mode managed \\
    --instance FineTune=ml.g6e.xlarge --instance SimEval=ml.g6e.xlarge \\
    --train-steps 200 --eval-trials 3 --threshold 0 --max-runtime-seconds 21600

Use --checkpoint-s3 to omit FineTune, or --through to stop after a named step.
Inspect supported choices and declared resources with {prog} cells --details.""",
    )
    run.add_argument("--deployment", help="Saved deployment name; default: arena.deployment_name in config.json")
    run.add_argument("--cell", help=f"Named model/simulator/suite combination from {prog} cells")
    run.add_argument("--model", help="Explicit model alternative to --cell: gr00t, molmoact2 or openvla")
    run.add_argument("--model-version", choices=["n16", "n17"],
                     help="Required with explicit --model gr00t; omit for other models")
    run.add_argument("--simulator", choices=["isaac_arena", "libero"],
                     help="Simulator for an explicit model selection")
    run.add_argument("--suite", help="Suite for an explicit selection; must match an exposed cell")
    run.add_argument("--mode", choices=["local", "managed"], required=True,
                     help="Use the prepared EC2 GPU host or managed SageMaker jobs")
    run.add_argument("--instance", action="append", default=[], metavar="STEP=TYPE",
                     help="Managed GPU instance; repeat for FineTune and SimEval when selected")
    run.add_argument("--volume-gb", action="append", default=[], metavar="STEP=GB",
                     help="Requested GPU-job volume override; defaults to cell declarations, "
                     "does not resize local disks")
    run.add_argument("--image", action="append", default=[], metavar="STEP=ECR_URI",
                     help="Published FineTune or SimEval image override; otherwise use saved images")
    run.add_argument("--sample", action="store_true",
                     help="Real short run: 200 training steps, 3 evaluation trials, threshold 0. "
                     "Explicit flags override sample defaults; checkpoint reuse still skips training.")
    run.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=None,
                     help="Record Arena evaluation video; config may select auto for supported cells")
    run.add_argument("--train-steps", type=int, help="Positive training-step count; required with FineTune")
    run.add_argument("--eval-trials", type=int,
                     help="Required with SimEval: episodes total for Arena, per task for LIBERO")
    run.add_argument("--eval-seed", type=int, help="Rollout seed (default: Arena 100, LIBERO 1000)")
    run.add_argument("--threshold", type=float,
                     help="Required with SuccessGate: success rate from 0 to 1; 0 checks workflow only")
    run.add_argument("--max-runtime-seconds", type=int, help="GPU job deadline; does not limit a status watcher")
    run.add_argument("--local-timeout-seconds", type=int,
                     help="Local worker total deadline, including preparation and all steps")
    run.add_argument("--image-pull-timeout-seconds", type=int,
                     help="Local per-image download allowance (default: 7200)")
    run.add_argument("--container-preparation-seconds", type=int,
                     help="Local Compose creation allowance (default: 600)")
    run.add_argument("--save-steps", type=int,
                     help="Training checkpoint interval (default: 1000000); "
                     "N1.6 requires at least --train-steps")
    run.add_argument("--checkpoint-s3",
                     help="Exact s3://bucket/key checkpoint object in a versioned bucket; "
                     "omit FineTune and training-only arguments")
    run.add_argument("--through", choices=["FineTune", "SimEval", "Validate", "SuccessGate", "RegisterModel"],
                     help="Run through this step, including its dependencies")
    run.add_argument("--run-id", help="Unique run name (1-40 letters, digits or hyphens); default generated")
    run.add_argument("--group", help="Optional label for listing a set of independent executions")
    run.add_argument("--dry-run", action="store_true", help="Print the resolved request without AWS writes")
    run.add_argument("--offline", action="store_true", help="With --dry-run, skip AWS prerequisite reads")
    run.add_argument("--wait", action="store_true", help="Follow after submission; Ctrl-C only stops following")
    run.add_argument("--watch-timeout-seconds", type=int, default=86400,
                     help="Following allowance; never cancels the workload (default: 86400)")

    status = commands.add_parser("status", help="List recorded operations, or inspect/follow one run")
    status.add_argument("id", nargs="?", help="Saved run or deployment name; omit to list operations")
    status.add_argument("--follow", action="store_true",
                        help="Watch the named operation; Ctrl-C does not cancel it")
    status.add_argument("--group", help="List operations carrying this group label")
    status.add_argument("--watch-timeout-seconds", type=int, default=86400,
                        help="Following allowance; never cancels the workload (default: 86400)")
    stop = commands.add_parser("stop", help="Cancel one recorded execution; preserve its host and evidence")
    stop.add_argument("id", help="Saved run name to cancel")
    cleanup = commands.add_parser("cleanup", help="Stop owned work; archive/remove verified local containers")
    cleanup.add_argument("id", help="Saved run name")
    cleanup.add_argument("--yes", action="store_true", help="Apply; otherwise print the cleanup plan")
    cleanup.add_argument("--timeout-seconds", type=int, default=3600,
                         help="Cancellation/archive allowance (default: 3600)")
    report = commands.add_parser("report", help="Verify and export the exact run's results and recordings")
    report.add_argument("id", help="Saved run ID")
    report.add_argument("--output-dir", help="Directory for this run's evidence and readable summary")
    report.add_argument("--include-video", action="store_true", help="Export this run's recorded videos when available")
    destroy = commands.add_parser("destroy", help="Preserve evidence and remove this deployment's owned resources")
    destroy.add_argument("id", help="Saved deployment name")
    destroy.add_argument("--yes", action="store_true", help="Execute the displayed teardown; otherwise print a plan")
    destroy.add_argument("--remove-evidence", action="store_true",
                         help="Verify a local archive before removing the temporary AWS archive bucket")
    return root


def catalog(args):
    from .registry import named_cells, resolve, resolve_suite

    cells = named_cells()
    if args.cell and args.cell not in cells:
        raise ValueError(f"Unknown cell {args.cell!r}. Available: {', '.join(cells)}")
    rows = []
    for name, cell in cells.items():
        if args.cell and args.cell != name:
            continue
        spec = resolve(cell["model"], cell["simulator"])
        suite = resolve_suite(cell["suite"])
        row = {"cell": name, **cell,
               "modes": ["managed", "local"] if cell["local_profile"] else ["managed"]}
        if args.details:
            row.update(
                declared_instances={"FineTune": spec.train_instance, "SimEval": spec.eval_instance},
                volume_gb={"FineTune": spec.train_volume_gb, "SimEval": spec.eval_volume_gb},
                task_count=len(suite.canonical_task_ids),
                episodes_meaning="total" if cell["simulator"] == "isaac_arena" else "per task",
                workflow=["FineTune", "SimEval", "Validate", "SuccessGate", "RegisterModel (managed)"],
                note="Declared hardware is not a measured minimum. See README execution evidence.",
            )
        rows.append(row)
    return rows


def show(value, as_json):
    if as_json:
        print(json.dumps(value, indent=2, default=str))
        return
    if isinstance(value, list):
        for row in value:
            if "modes" in row:
                print(f"{row['cell']:24} {', '.join(row['modes']):15} {row['suite']}")
                if "declared_instances" in row:
                    print(f"  GPU declarations: {row['declared_instances']}; volumes: {row['volume_gb']}")
                    print(f"  Episodes: {row['episodes_meaning']}; tasks: {row['task_count']}")
                    print(f"  {row['note']}")
            else:
                print(f"{row.get('kind', 'run'):11} {row['id']:32} {row.get('status', 'Unknown')}")
                if row.get("status") == "UnreadableRecord":
                    print(f"  {row['record_path']}: {row['failure_reason']}")
        if not value:
            print("No saved operations.")
        return
    if "parameters" in value:
        p = value["parameters"]
        print(f"Cell: {value['cell']} | mode: {value['mode']} | deployment: {value['deployment']}")
        print(f"Account: {value['account_id']} | application region: {value['region']}")
        if value.get("local_target"):
            target = value["local_target"]
            print(f"EC2 host: {target['instance_id']} in {target['host_region']}; "
                  f"type: {target.get('instance_type', 'not observed')}; "
                  f"development bucket: {target['development_bucket']}")
        print("Steps: " + " → ".join(value["steps"]))
        for step, key in (("FineTune", "TrainInstanceType"), ("SimEval", "EvalInstanceType")):
            if step in value["steps"]:
                print(f"  {step}: {p[key]}")
        for key in ("TrainSteps", "EvalTrials", "EvalSeed", "SuccessThreshold", "MaxRuntimeSeconds"):
            if key in p:
                print(f"  {key}: {p[key]}")
        if value.get("local_limits"):
            print(f"  Local allowances (seconds): {value['local_limits']}")
        for step, uri in value["images"].items():
            if step in value["steps"]:
                print(f"  {step} image: {uri}")
        if value.get("checkpoint_s3"):
            print(f"FineTune omitted; checkpoint: {value['checkpoint_s3']}")
        print(value["checks"])
        return
    print(f"{value['id']}: {value.get('status', 'Unknown')}")
    if value.get("requested_steps"):
        print("  Requested steps: " + " → ".join(value["requested_steps"]))
        if not value.get("complete_training_workflow"):
            print("  Scope: selected steps only; the full training pipeline was not requested")
    if value.get("elapsed_seconds") is not None:
        print(f"  Elapsed: {value['elapsed_seconds'] / 60:.1f} minutes")
    for key in ("activity", "transport_activity", "failure_reason", "next_action", "execution_arn",
                "model_package_arn", "run_dir", "receipt_uri",
                "selection_uri", "selection_version"):
        if value.get(key):
            print(f"  {key}: {value[key]}")
    if value.get("last_log_age_seconds") is not None:
        print(f"  Last log activity: {value['last_log_age_seconds']} seconds ago")
    if value.get("service_state"):
        print(f"  Host service: {value['service_state']}")
    for step, output in value.get("outputs", {}).items():
        print(f"  {step} output: {output['uri']} (S3 version {output['version_id']})")
    if "requested_steps_succeeded" in value:
        print(f"  Requested steps succeeded: {value['requested_steps_succeeded']}")
    if value.get("verification_status"):
        print(f"  Requested-run verification: {value['verification_status']}")
    elif "independently_verified" in value:
        print(f"  Requested-run verification: "
              f"{'passed' if value['independently_verified'] else 'not established'}")
    if value.get("verification_scope"):
        print(f"  Verification scope: {value['verification_scope']}")
    if value.get("episodes") is not None:
        print(f"  Evaluation episodes: {value['episodes']}; success rate: {value.get('success_rate')}")
        print("  A workflow sample with a zero threshold does not establish model quality.")
    if value.get("cleanup"):
        print("  Cleanup: " + json.dumps(value["cleanup"], default=str))
    if value.get("readiness_checks"):
        checks = value["readiness_checks"]
        print("  Current deployment checks:")
        print(f"    PASS  AWS account: {checks['configured_account']}")
        print(f"    {checks['published_images'].upper()}  Published images: {checks['published_image_count']}")
        if checks.get("instance_id"):
            print(f"    EC2 {checks['instance_id']} ({checks['instance_type']}): "
                  f"{checks['instance_state']}; SSM {checks['ssm_status']}")
        host = checks.get("host", {})
        if host:
            print(f"    PASS  Checkout: {host['source_commit']}")
            print(f"    GPU: {host['gpu']}")
            for name, disk in host["disks"].items():
                print(f"    {'PASS' if disk['passed'] else 'FAIL'}  {name} disk: "
                      f"{disk['free_gib']:.2f} GiB free / {disk['required_gib']} GiB sample reserve")
            print(f"    {'PASS' if host['scratch_is_separate'] else 'FAIL'}  Scratch uses a separate filesystem")
            missing = [uri for uri, identity in host["images"].items() if not identity]
            print(f"    {'PASS' if not missing else 'FAIL'}  Selected images present on GPU host")
            print(f"    Sample readiness: {'passed' if host['sample_ready'] else 'not ready'}")
    if value.get("absence_checks"):
        print("  Fresh AWS removal checks:")
        for name, values in value["absence_checks"].items():
            print(f"    PASS  {name.replace('_', ' ')}: {values}")
    if value.get("local_evidence"):
        print("  Verified local evidence: " + json.dumps(value["local_evidence"], default=str))
    for name in ("retained_archive", "removed_archive", "report_directory"):
        if value.get(name):
            print(f"  {name.replace('_', ' ')}: {value[name]}")
    if value.get("status") == "TeardownPlan":
        print("  Owned resources:", value["owned_resources"])
        print("  Recorded runs:", ", ".join(value["runs"]) or "none")
        print("  Steps:", " → ".join(value["steps"]))
        print("  Retains:", value["retains"])


def main(argv=None, *, prog="vla"):
    with command_context(prog):
        return _main(argv)


def _main(argv):
    command = parser()
    args = command.parse_args(argv)
    store = Store(args.state_dir)
    try:
        from .cli_settings import apply_settings
        apply_settings(args)
        if getattr(args, "watch_timeout_seconds", 1) <= 0:
            raise ValueError("--watch-timeout-seconds must be positive")
        if args.command == "cells":
            show(catalog(args), args.json)
        elif args.command == "deploy":
            if args.create_local_host and any((args.local_host, args.use_existing, args.prepare_host)):
                raise ValueError("--create-local-host cannot be combined with a supplied host or existing selection")
            if args.gpu_zone and not args.create_local_host and not args.resume:
                raise ValueError("--gpu-zone requires --create-local-host")
            if args.resume and any((args.create_local_host, args.gpu_zone, args.hf_token_file, args.ngc_token_file)):
                raise ValueError("--resume uses the saved request; omit new host/token choices")
            if args.use_existing and any((args.prepare_host, args.resume, args.plan, args.yes,
                                          args.environment, args.hf_secret_name, args.ngc_secret_name,
                                          args.input_s3_arn, args.existing_ecr_repo,
                                          args.hf_token_file, args.ngc_token_file)):
                raise ValueError("--use-existing is selection only; omit provisioning/host-preparation flags")
            if args.resume and any((args.prepare_host, args.cell, args.image, args.selection,
                                    args.local_host, args.host_region, args.development_bucket,
                                    args.expected_role, args.scratch_root, args.project, args.environment,
                                    args.hf_secret_name, args.ngc_secret_name, args.input_s3_arn,
                                    args.existing_ecr_repo)):
                raise ValueError("--resume uses the saved request; omit new preparation choices")
            if args.prepare_host and any((args.existing_ecr_repo, args.hf_token_file, args.ngc_token_file)):
                raise ValueError("Repository ownership and token files apply only to a new infrastructure deployment")
            with contextlib.redirect_stdout(sys.stderr):
                if args.use_existing:
                    from .deployment import select_existing
                    with store.lock("deployments", args.name):
                        result = select_existing(args, store)
                else:
                    from .provisioning import deploy
                    result = deploy(args, store)
            show(result, args.json)
            if not args.json:
                print(f"Operation record: {store.path('deployments', args.name)}")
        elif args.command == "run":
            from .execution import follow, prepare, submit
            from .launch_request import resolve_run
            request = resolve_run(args)
            if args.offline and not args.dry_run:
                raise ValueError("--offline requires --dry-run")
            deployment = store.load("deployments", args.deployment)
            if args.configured_account and deployment["account_id"] != args.configured_account:
                raise ValueError("Saved deployment account differs from arena.account_id in config.json")
            with contextlib.redirect_stdout(sys.stderr):
                request = prepare(request, deployment, check_remote=not args.offline)
            if args.dry_run:
                show(request, args.json)
                return 0
            if not args.json:
                show(request, False)
            with contextlib.redirect_stdout(sys.stderr):
                result = submit(request, deployment, store, args.run_id)
            show(result, args.json)
            if not args.json:
                print(f"Inspect: {store.follow_command(result['id'])}")
            if args.wait:
                return follow(result, as_json=args.json, timeout_seconds=args.watch_timeout_seconds,
                              store=store)
        elif args.command == "status":
            if args.id and args.group:
                raise ValueError("Select an operation ID or --group, not both")
            if args.id:
                from .execution import follow, refresh
                if store.path("runs", args.id).exists():
                    record = store.load("runs", args.id)
                    if args.follow:
                        return follow(record, as_json=args.json,
                                      timeout_seconds=args.watch_timeout_seconds, store=store)
                    with contextlib.redirect_stdout(sys.stderr):
                        current = refresh(record, store)
                    show(current, args.json)
                else:
                    if args.follow:
                        from .deployment_status import follow_deployment
                        return follow_deployment(store, args.id, as_json=args.json,
                                                 timeout_seconds=args.watch_timeout_seconds)
                    deployment = store.load("deployments", args.id)
                    if deployment.get("lifecycle"):
                        from .lifecycle import inspect
                        with contextlib.redirect_stdout(sys.stderr):
                            deployment = inspect(deployment, store)
                    show(deployment, args.json)
            else:
                if args.follow:
                    raise ValueError("--follow requires a run ID")
                rows = store.list()
                if args.group:
                    rows = [row for row in rows if row.get("group") == args.group]
                show(rows, args.json)
        elif args.command == "stop":
            from .execution import stop
            record = store.load("runs", args.id)
            with contextlib.redirect_stdout(sys.stderr):
                record = stop(record, store=store)
            if args.json:
                show(record, True)
            else:
                print(f"Cancellation requested for {args.id}; inspect with "
                      f"{command_name()} status {args.id}")
        elif args.command == "cleanup":
            from .cleanup import cleanup
            with contextlib.redirect_stdout(sys.stderr):
                result = cleanup(store.load("runs", args.id), store, execute=args.yes,
                                 timeout_seconds=args.timeout_seconds)
            if args.yes:
                show(result, args.json)
            else:
                print(json.dumps(result, indent=2))
        elif args.command == "report":
            from .reporting import report
            with contextlib.redirect_stdout(sys.stderr):
                result = report(store.load("runs", args.id), store, output_dir=args.output_dir,
                                include_video=args.include_video)
            show(result, args.json)
        elif args.command == "destroy":
            from .lifecycle import destroy
            with contextlib.redirect_stdout(sys.stderr):
                result = destroy(store.load("deployments", args.id), store, execute=args.yes,
                                 remove_evidence=args.remove_evidence)
            show(result, args.json)
        return 0
    except KeyboardInterrupt:
        if args.command == "deploy":
            print(f"Preparation interrupted; submitted builds/host commands may still be running. "
                  f"Inspect {command_name()} status {args.name}, then continue with "
                  f"{command_name()} deploy --name {args.name} --resume.", file=sys.stderr)
        else:
            print("Stopped watching; no workload cancellation was requested.", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            import traceback
            traceback.print_exc(file=sys.stderr)
        else:
            print(f"{command_name()}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def module_main():
    """Retain the existing module-only pipeline-definition export."""
    if sys.argv[1:2] != ["definition"]:
        return main()
    legacy = argparse.ArgumentParser(prog="python -m vla_pipeline.cli definition")
    legacy.add_argument("-o", "--out")
    args = legacy.parse_args(sys.argv[2:])
    try:
        from .config import load_config
        from .pipeline import build_pipeline
        with contextlib.redirect_stdout(sys.stderr):
            body = json.loads(build_pipeline(load_config()).definition())
        pretty = json.dumps(body, indent=2)
        if args.out:
            from pathlib import Path
            Path(args.out).write_text(pretty + "\n")
            print(f"wrote {args.out} ({len(pretty)} bytes)")
        else:
            print(pretty)
        return 0
    except Exception as exc:
        print(f"definition: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(module_main())
