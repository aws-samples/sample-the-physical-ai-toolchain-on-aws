"""Native teardown must retain other deployments and reject unknown resource ownership."""
import copy
from datetime import datetime, timezone

import boto3
from botocore.stub import Stubber
import pytest

from vla_pipeline.lifecycle import DeploymentEnvironment, require_owner_tag
from vla_pipeline.operations import OperationBusy, Store


def deployment(tmp_path):
    store = Store(tmp_path / "state")
    record = {
        "id": "example", "project": "example", "source_commit": "a" * 40,
        "work_directory": str(tmp_path / "work"), "request": {}, "status": "Ready",
        "account_id": "111122223333", "region": "us-east-1", "profile": None,
    }
    store.save("deployments", record)
    return DeploymentEnvironment(record, store), store


def test_teardown_collects_all_own_runs_and_leaves_other_deployments_out(tmp_path):
    env, store = deployment(tmp_path)
    own = {"id": "my-run", "deployment": "example", "mode": "managed",
           "account_id": "111122223333", "region": "us-east-1"}
    store.create_run(own)
    store.create_run({**own, "id": "someone-elses-run", "deployment": "another"})
    assert [run["id"] for run in env.runs()] == ["my-run"]
    assert "controller" not in env.states
    assert env.deployment_path == store.path("deployments", "example")


def test_missing_registered_run_refuses_to_shrink_teardown_scope(tmp_path):
    env, _ = deployment(tmp_path)
    env.deployment["run_ids"] = ["missing-run"]
    with pytest.raises(ValueError, match="No saved run"):
        env.runs()


def test_moved_run_cannot_change_deployment_account(tmp_path):
    env, store = deployment(tmp_path)
    store.create_run({"id": "wrong-account", "deployment": "example", "mode": "managed",
                      "account_id": "999988887777", "region": "us-east-1"})
    with pytest.raises(ValueError, match="ownership differs"):
        env.runs()


@pytest.mark.parametrize("actual_token", ["another-owner", None])
def test_resource_tag_must_match_before_deletion(actual_token):
    # These are SDK response tests, not live AWS deletion tests.
    sm = boto3.client("sagemaker", region_name="us-east-1",
                      aws_access_key_id="test", aws_secret_access_key="test")
    arn = "arn:aws:sagemaker:us-east-1:111122223333:pipeline/my-run"
    tags = [{"Key": "pai-arena-owner", "Value": actual_token}] if actual_token else []
    with Stubber(sm) as stub:
        stub.add_response("list_tags", {"Tags": tags}, {"ResourceArn": arn})
        with pytest.raises(ValueError, match="refusing deletion"):
            require_owner_tag(sm, arn, "expected-owner")
        stub.assert_no_pending_responses()


def test_recorded_managed_execution_must_match_its_run_pipeline(tmp_path):
    env, store = deployment(tmp_path)
    record = {"id": "my-run", "deployment": "example", "mode": "managed",
              "account_id": "111122223333", "region": "us-east-1",
              "execution_arn": "arn:aws:sagemaker:us-east-1:111122223333:pipeline/vla-my-run/execution/abc",
              "pipeline_name": "vla-my-run"}
    store.create_run(record)
    assert env.managed_executions() == {
        record["execution_arn"]: "arn:aws:sagemaker:us-east-1:111122223333:pipeline/vla-my-run"}
    changed = copy.deepcopy(record)
    changed["pipeline_name"] = "another-pipeline"
    store.save("runs", changed)
    with pytest.raises(ValueError, match="ownership record"):
        env.managed_executions()


def test_teardown_cannot_lock_a_deployment_during_submission(tmp_path, monkeypatch):
    from vla_pipeline import execution

    env, store = deployment(tmp_path)
    env.persist()
    request = {"cell": "gr00t-n16-arena", "mode": "managed",
               "steps": ["FineTune"], "parameters": {"TrainSteps": 200}}
    monkeypatch.setattr(execution, "source_identity", lambda: "a" * 40)

    def submit_without_aws(request, selected, record, directory, operation_store):
        # Exercise the real filesystem locks while standing in for the AWS
        # submission call. A teardown must be excluded during this interval.
        with pytest.raises(OperationBusy):
            with operation_store.lock("deployments", selected["id"]):
                pytest.fail("Teardown acquired a deployment that is still submitting")
        record.update(status="Submitted", execution_arn="recorded-test-execution")

    monkeypatch.setattr(execution, "submit_managed", submit_without_aws)
    result = execution.submit(request, env.deployment, store, "atomic-run")
    assert result["status"] == "Submitted"
    with store.lock("deployments", env.name):
        assert store.load("deployments", env.name)["run_ids"] == ["atomic-run"]
        assert store.load("runs", "atomic-run")["execution_arn"] == "recorded-test-execution"


@pytest.mark.parametrize("confirmed,actual_owner,has_packages,outcome", [
    (True, "expected-owner", False, "removed"),
    (True, "expected-owner", True, "retained"),
    (False, "another-owner", False, "retained"),
    (True, "another-owner", False, "refused"),
])
def test_model_group_cleanup_preserves_shared_or_unowned_resources(
        tmp_path, monkeypatch, confirmed, actual_owner, has_packages, outcome):
    from vla_pipeline import lifecycle

    env, _ = deployment(tmp_path)
    name = "vla-arena-gr00t"
    arn = "arn:aws:sagemaker:us-east-1:111122223333:model-package-group/" + name
    run = {"resource_owner_token": "expected-owner",
           "created_model_group": {"name": name, "arn": arn, "confirmed": confirmed}}
    sm = boto3.client("sagemaker", region_name="us-east-1",
                      aws_access_key_id="test", aws_secret_access_key="test")
    monkeypatch.setattr(env, "runs", lambda: [run])
    monkeypatch.setattr(env, "client", lambda _: sm)
    # Test the native group lifecycle after the shared pipeline/image retirement.
    monkeypatch.setattr(lifecycle.environment_cleanup, "retire_outputs", lambda _: None)
    monkeypatch.setattr(lifecycle, "model_group_exists", lambda *_: True)
    with Stubber(sm) as stub:
        stub.add_response("list_tags", {"Tags": [{"Key": "pai-arena-owner", "Value": actual_owner}]},
                          {"ResourceArn": arn})
        if actual_owner == "expected-owner":
            packages = [{
                "ModelPackageGroupName": name, "ModelPackageVersion": 1,
                "ModelPackageArn": arn.replace("model-package-group/", "model-package/") + "/1",
                "CreationTime": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "ModelPackageStatus": "Completed", "ModelApprovalStatus": "PendingManualApproval",
            }] if has_packages else []
            stub.add_response("list_model_packages", {"ModelPackageSummaryList": packages},
                              {"ModelPackageGroupName": name})
        if outcome == "removed":
            stub.add_response("delete_model_package_group", {}, {"ModelPackageGroupName": name})
        if outcome == "refused":
            with pytest.raises(ValueError, match="refusing deletion"):
                env.retire_outputs()
        else:
            env.retire_outputs()
            receipt = lifecycle.load(env.root / "teardown/native-resources.json")
            assert (name in receipt["groups"]) == (outcome == "removed")
            assert (name in receipt["retained_groups"]) == (outcome == "retained")
        stub.assert_no_pending_responses()
