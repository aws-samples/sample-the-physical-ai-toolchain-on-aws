"""Provision and retire resources using Arena's deployment and run records."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import uuid

from .resources.environment import Environment, load, save, support
from .resources import environment_archive, environment_cleanup, environment_deploy

from .operations import timestamp


class DeploymentEnvironment(Environment):
    """Supply deployment ownership and run records to resource operations."""

    def __init__(self, deployment, store):
        self.deployment = deployment
        self.store = store
        self.name = deployment["id"]
        self.project = deployment["project"]
        self.root = Path(deployment["work_directory"]) / "lifecycle"
        self.root.mkdir(parents=True, exist_ok=True)
        request = deployment["request"]
        self.record = deployment.setdefault("lifecycle", {
            "candidate": deployment["source_commit"], "status": "Preparing", "phases": {},
            "config": {
                "account_id": deployment["account_id"], "region": deployment["region"],
                "profile": deployment.get("profile"), "gpu_instance": request.get("create_local_host"),
                "gpu_zone": request.get("gpu_zone"),
                **{key: request.get(key) for key in
                   ("hf_secret_name", "ngc_secret_name", "hf_token_file", "ngc_token_file")},
            },
        })
        self._session = None

    @property
    def states(self):
        work = Path(self.deployment["work_directory"])
        return {"foundation": work / "foundation", "arena": work / "component",
                "gpu": self.root / "gpu", "secrets": self.root / "secrets"}

    @property
    def deployment_path(self):
        return self.store.path("deployments", self.name)

    @property
    def notebook_path(self):
        from .execution import COMPONENT
        return COMPONENT / "notebooks/runbook.ipynb"

    @property
    def evidence_roots(self):
        return {
            "deployment": self.deployment_path.parent,
            **{"runs/" + run["id"]: self.store.path("runs", run["id"]).parent for run in self.runs()},
        }

    @property
    def host_policy_name(self):
        return "VlaCli-" + self.name

    def persist(self):
        self.record["updated_at"] = timestamp()
        self.store.save("deployments", self.deployment)

    def runs(self):
        found = {r["id"]: r for r in self.store.list()
                 if r["kind"] == "run" and r.get("deployment") == self.name}
        # Missing registered records must not silently shrink the teardown scope.
        for run_id in self.deployment.get("run_ids", []):
            found[run_id] = self.store.load("runs", run_id)
        for record in found.values():
            if (record["deployment"], record["account_id"], record["region"]) != (
                    self.name, self.config["account_id"], self.config["region"]):
                raise ValueError("Run ownership differs from this deployment")
        return list(found.values())

    def cleanup_reports(self):
        return [
            {"id": r["id"], "scenario": r["mode"], "observed": r,
             "acceptance": "passed" if r.get("independently_verified") else "not_passed"}
            for r in self.runs()
        ]

    def managed_executions(self):
        result = {}
        for run in self.runs():
            if run["mode"] != "managed" or not run.get("execution_arn"):
                continue
            expected = f"vla-{run['id']}"
            if run.get("pipeline_name") != expected:
                raise ValueError("Run pipeline name differs from its native ownership record")
            result[run["execution_arn"]] = (
                f"arn:aws:sagemaker:{self.config['region']}:{self.config['account_id']}:pipeline/{expected}")
        return result

    def host_capture_context(self):
        local = self.deployment.get("local", {})
        if not local.get("remote_ready"):
            return None
        roots = [local["state_dir"]]
        roots += [r["run_dir"] for r in self.runs() if r["mode"] == "local" and r.get("run_dir")]
        return {
            "gpu": {"development_bucket": local["development_bucket"]},
            "targets": {"gpu": (local["instance_id"], local["component"] + "/.venv/bin/python", roots)},
        }

    def finish_workloads(self):
        from .cleanup import cleanup
        from .execution import refresh
        environment_cleanup.inventory(self)
        for build in self.deployment.get("builds", {}).values():
            if build.get("status") == "Submitting" and not build.get("id"):
                from .image_builds import reconcile_build
                reconcile_build(self.client("codebuild"), build)
                self.persist()
                if not build.get("id"):
                    raise ValueError("An image build submission needs reconciliation before teardown")
        for run in self.runs():
            if run.get("cleanup", {}).get("status") == "Complete":
                continue
            if run["mode"] == "managed" and not run.get("execution_arn"):
                run = refresh(run, self.store)
                if not run.get("execution_arn"):
                    if run.get("client_request_token"):
                        raise ValueError("Managed submission still needs reconciliation before teardown: " + run["id"])
                    # submit_managed records a token before calling StartPipeline.
                    # With no token, only source/group/pipeline preparation occurred.
                    run["cleanup"] = {"status": "Complete", "finished_at": timestamp(),
                                      "scope": "Submission ended before any managed execution request"}
                    self.store.save("runs", run)
                    continue
            result = cleanup(run, self.store, execute=True)
            if result.get("cleanup", {}).get("status") != "Complete":
                raise ValueError("Workload cleanup did not finish for " + run["id"])

    def retire_outputs(self):
        environment_cleanup.retire_outputs(self)
        sm = self.client("sagemaker")
        path = self.root / "teardown/native-resources.json"
        receipts = load(path) if path.exists() else {"pipelines": [], "groups": []}
        retained = receipts.setdefault("retained_groups", {})
        for run in self.runs():
            token = run.get("resource_owner_token")
            if not token:
                continue
            # A CreatePipeline/Start response may have been lost. Its unique
            # ownership tag and empty execution list are required before deleting
            # a pipeline without a recorded execution.
            if run.get("pipeline_creation_requested") and not run.get("execution_arn"):
                name = run["pipeline_name"]
                arn = f"arn:aws:sagemaker:{run['region']}:{run['account_id']}:pipeline/{name}"
                if environment_cleanup.pipeline_exists(sm, arn):
                    require_owner_tag(sm, arn, token)
                    executions = [r for page in sm.get_paginator("list_pipeline_executions").paginate(
                        PipelineName=name) for r in page["PipelineExecutionSummaries"]]
                    if executions:
                        raise ValueError("An execution exists without a saved response. Reconcile " + arn)
                    sm.delete_pipeline(PipelineName=name)
                if arn not in receipts["pipelines"]:
                    receipts["pipelines"].append(arn)
                    save(path, receipts)
            group = run.get("created_model_group")
            if group:
                if model_group_exists(sm, group["name"]):
                    try:
                        require_owner_tag(sm, group["arn"], token)
                    except ValueError:
                        if group.get("confirmed"):
                            raise
                        retained[group["name"]] = "Creation was not confirmed; this group belongs to another owner"
                        save(path, receipts)
                        continue
                    packages = [r for page in sm.get_paginator("list_model_packages").paginate(
                        ModelPackageGroupName=group["name"]) for r in page["ModelPackageSummaryList"]]
                    if packages:
                        retained[group["name"]] = "Group now contains other executions' model packages"
                        save(path, receipts)
                        continue
                    sm.delete_model_package_group(ModelPackageGroupName=group["name"])
                if group["name"] not in receipts["groups"]:
                    receipts["groups"].append(group["name"])
                    save(path, receipts)
        save(path, receipts)

    def verify_native_absence(self):
        path = self.root / "teardown/native-resources.json"
        if not path.exists():
            raise ValueError("Native resource retirement has not completed")
        receipts, sm = load(path), self.client("sagemaker")
        if any(environment_cleanup.pipeline_exists(sm, arn) for arn in receipts["pipelines"]):
            raise ValueError("A recorded pipeline without an execution remains")
        if any(model_group_exists(sm, name) for name in receipts["groups"]):
            raise ValueError("A created model package group remains")
        return receipts


def require_owner_tag(sm, arn, token):
    tags = {item["Key"]: item["Value"] for page in sm.get_paginator("list_tags").paginate(
        ResourceArn=arn) for item in page["Tags"]}
    if tags.get("pai-arena-owner") != token:
        raise ValueError("Resource ownership tag differs; refusing deletion: " + arn)


def model_group_exists(sm, name):
    return any(row["ModelPackageGroupName"] == name
               for page in sm.get_paginator("list_model_package_groups").paginate(NameContains=name)
               for row in page["ModelPackageGroupSummaryList"])


def prepare_environment(deployment, store):
    from .resources.environment_setup import check_capacity, token_secrets
    env = DeploymentEnvironment(deployment, store)
    baseline = environment_deploy.record_baseline(env)
    supplied = set(deployment["request"].get("existing_ecr_repos", []))
    supplied.update(baseline["external_ecr"])
    deployment["request"]["existing_ecr_repos"] = sorted(supplied)
    env.persist()
    if deployment["request"].get("create_local_host"):
        env.phase("capacity-check", check_capacity)
    env.phase("token-secrets", token_secrets)
    return env


def create_host(deployment, store):
    env = DeploymentEnvironment(deployment, store)
    env.phase("gpu-host", environment_deploy.gpu_host)
    gpu = load(env.root / "gpu-outputs.json")
    deployment["owned_gpu"] = gpu
    deployment["local"].update(
        instance_id=gpu["instance_id"], host_region=deployment["region"],
        expected_role=gpu["runtime_role"], development_bucket=gpu["development_bucket"],
        scratch_root="/opt/dlami/nvme/vla-tests", host_supplied=False,
    )
    env.persist()


def destroy(deployment, store, *, execute=False, remove_evidence=False):
    if not deployment.get("lifecycle") or not deployment.get("infrastructure_owned"):
        raise ValueError("This selection has no owned deployment lifecycle. Use cleanup RUN_ID for its runs; "
                         "supplied infrastructure remains with its original owner.")
    env = DeploymentEnvironment(deployment, store)
    if not execute:
        owned = support.terraform_inventory(env.states)
        return {"id": env.name, "status": "TeardownPlan", "runs": [r["id"] for r in env.runs()],
                "owned_resources": {name: len(state["resources"]) for name, state in owned.items()},
                "retains": "Verified local evidence and supplied/shared resources",
                "steps": environment_cleanup.PHASES}
    with store.lock("deployments", env.name):
        deployment = store.load("deployments", env.name)
        env = DeploymentEnvironment(deployment, store)
        env.client("sts")
        if deployment["status"] == "Destroyed":
            return inspect(deployment, store)
        if "remove_evidence" in env.record and env.record["remove_evidence"] != remove_evidence:
            raise ValueError("Resume teardown with its original --remove-evidence choice")
        deployment["status"] = "Destroying"
        env.record.update(status="Destroying", remove_evidence=remove_evidence)
        env.persist()
        try:
            for name, action in environment_cleanup.actions():
                if name == "finish-workloads":
                    action = lambda e: e.finish_workloads()
                elif name == "retire-outputs":
                    action = lambda e: e.retire_outputs()
                elif name == "verify-teardown":
                    def action(e):
                        e.verify_native_absence()
                        environment_cleanup.verify_teardown(e)
                env.phase(name, action)
        except BaseException as exc:
            deployment.update(failure_reason=f"{type(exc).__name__}: {exc}",
                              activity="Teardown incomplete; evidence retained. Repeat the same destroy command.")
            env.persist()
            raise
        deployment.update(status="Destroyed", activity="Owned resources removed; evidence preserved",
                          completed_at=timestamp())
        env.record["status"] = "Destroyed"
        env.persist()
    return inspect(deployment, store)


def inspect(deployment, store):
    """Observe actual resources without treating a saved Ready/Destroyed label as proof."""
    env = DeploymentEnvironment(deployment, store)
    result = dict(deployment)
    if deployment["status"] == "Destroyed":
        result["absence_checks"] = environment_cleanup.verify_absence(env)
        result["absence_checks"]["native_resources"] = env.verify_native_absence()
        if env.record.get("remove_evidence"):
            closure = load(env.root / "result.json")
            environment_archive.check_removed(env, closure["removed_archive"])
            result["local_evidence"] = environment_archive.verify_local(env)
            result["removed_archive"] = closure["removed_archive"]
        else:
            result["retained_archive"] = load(env.root / "result.json")["retained_archive"]
        return result
    from .deployment import image_digest
    session = _session(env)
    images = sorted({uri for images in deployment.get("images", {}).values() for uri in images.values()})
    for uri in images:
        if image_digest(session, uri) != uri:
            raise ValueError("Published image differs from the selected digest")
    checks = {"configured_account": env.config["account_id"], "published_image_count": len(images),
              "published_images": "verified" if images else "not prepared"}
    local = deployment.get("local", {})
    if local.get("instance_id"):
        ec2 = session.client("ec2", region_name=local["host_region"])
        ssm = session.client("ssm", region_name=local["host_region"])
        rows = ec2.describe_instances(InstanceIds=[local["instance_id"]])["Reservations"]
        host = rows[0]["Instances"][0]
        online = ssm.describe_instance_information(
            Filters=[{"Key": "InstanceIds", "Values": [local["instance_id"]]}])["InstanceInformationList"]
        checks.update(instance_id=host["InstanceId"], instance_state=host["State"]["Name"],
                      instance_type=host["InstanceType"],
                      ssm_status=online[0]["PingStatus"] if online else "not online")
        if local.get("remote_ready") and host["State"]["Name"] == "running" and (
                online and online[0]["PingStatus"] == "Online"):
            from .remote_commands import submit, wait
            receipt = {}
            path = Path(deployment["work_directory"]) / "observations" / (uuid.uuid4().hex + ".json")
            command = shlex.join([
                local["component"] + "/.venv/bin/python", "-m", "vla_pipeline.remote_worker", "inspect",
                "--state-dir", local["state_dir"], "--deployment", deployment["id"]])
            save_receipt = lambda: save(path, receipt)
            submit(ssm, local["instance_id"], receipt, [command], save_receipt, timeout=300)
            reply = wait(ssm, local["instance_id"], receipt, save_receipt)
            observations = [line.removeprefix("VLA_HOST_INSPECTION=")
                            for line in reply["StandardOutputContent"].splitlines()
                            if line.startswith("VLA_HOST_INSPECTION=")]
            if len(observations) != 1:
                raise ValueError("Host readiness response is missing; inspect " + str(path))
            checks["host"] = json.loads(observations[0])
    result["readiness_checks"] = checks
    return result


def _session(env):
    env.client("sts")
    return env._session
