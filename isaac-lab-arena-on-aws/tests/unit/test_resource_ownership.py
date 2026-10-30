"""Resource cleanup must reject changes beyond the captured ownership and archive."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import boto3
from botocore.exceptions import ClientError
from botocore.stub import Stubber
import pytest

from vla_pipeline.resources.environment import support
from vla_pipeline.resources import environment_cleanup
from vla_pipeline.resources.environment_cleanup import (
    allow_owned_repository_deletion, wait_for_secret_absence,
)


def test_destroy_plan_accepts_only_the_recorded_resource_id():
    recorded = {"resources": [{
        "address": "aws_instance.host", "attributes": {"id": "i-owned"},
    }]}
    plan = {"resource_changes": [{
        "mode": "managed", "address": "aws_instance.host",
        "change": {"actions": ["delete"], "before": {"id": "i-owned"}},
    }]}
    assert support.review_destroy_plan(plan, recorded) == ["aws_instance.host"]
    for field, changed in [
        ("address", "aws_instance.other"),
        ("change", {"actions": ["create"], "before": None}),
        ("change", {"actions": ["delete", "create"], "before": {"id": "i-owned"}}),
        ("change", {"actions": ["delete"], "before": {"id": "i-someone-else"}}),
    ]:
        unsafe = deepcopy(plan)
        unsafe["resource_changes"][0][field] = changed
        with pytest.raises(ValueError):
            support.review_destroy_plan(unsafe, recorded)


def test_partial_deletion_resumes_without_deleting_new_object_versions():
    item = {"key": "model.tar.gz", "version": "version-one", "bytes": 123, "etag": "etag"}
    frozen = {"objects": [item], "delete_markers": [{"key": "old", "version": "marker-one"}]}
    remaining = {"objects": [item], "delete_markers": []}
    assert support.exact_version_removals(remaining, frozen) == [
        {"Key": "model.tar.gz", "VersionId": "version-one"}]
    for field, changed in [("version", "version-two"), ("bytes", 124), ("etag", "changed")]:
        with pytest.raises(ValueError, match="Unarchived"):
            support.exact_version_removals(
                {"objects": [{**item, field: changed}], "delete_markers": []}, frozen)
    with pytest.raises(ValueError, match="unrecorded delete marker"):
        support.exact_version_removals(
            {"objects": [], "delete_markers": [{"key": "old", "version": "marker-two"}]}, frozen)


@pytest.mark.parametrize("existing,images,force_delete,allowed", [
    (False, [], False, True),
    (False, [{"imageDigest": "sha256:" + "a" * 64}], False, False),
    (True, [], False, False),
    (False, [], True, False),
])
def test_repository_protection_is_removed_only_for_new_empty_owned_repos(
        tmp_path, existing, images, force_delete, allowed):
    name = "vla/gr00t"
    support.save(tmp_path / "baseline.json", {"external_ecr": {name: []} if existing else {}})
    ecr = boto3.client("ecr", region_name="us-east-1",
                       aws_access_key_id="test", aws_secret_access_key="test")
    env = SimpleNamespace(root=tmp_path, states={"arena": tmp_path / "component"},
                          client=lambda _: ecr)
    owned = {"arena": {"resources": [{
        "address": f'aws_ecr_repository.repos["{name}"]', "type": "aws_ecr_repository",
        "attributes": {"name": name, "force_delete": force_delete},
    }]}}
    override = env.states["arena"] / "retire_owned_override.tf.json"
    with Stubber(ecr) as stub:
        if not existing and not force_delete:
            # The service model permits an omitted result list for an empty page.
            stub.add_response("list_images", {"imageIds": images} if images else {},
                              {"repositoryName": name})
        if allowed:
            allow_owned_repository_deletion(env, owned)
            assert support.load(override)["resource"]["aws_ecr_repository"]["repos"] == {
                "lifecycle": {"prevent_destroy": False}}
        else:
            with pytest.raises(ValueError):
                allow_owned_repository_deletion(env, owned)
            assert not override.exists()
        stub.assert_no_pending_responses()


@pytest.mark.parametrize("case", ["already_absent", "eventually_absent", "timeout", "not_deleted", "forbidden"])
def test_secret_teardown_requires_confirmed_absence(monkeypatch, case):
    client = boto3.client("secretsmanager", region_name="us-east-1",
                          aws_access_key_id="test", aws_secret_access_key="test")
    arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:owned-abcdef"
    request = {"SecretId": arn}
    elapsed = [0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        elapsed[0] += seconds

    monkeypatch.setattr(environment_cleanup, "time",
                        SimpleNamespace(monotonic=lambda: elapsed[0], sleep=sleep))
    deleting = {"ARN": arn, "DeletedDate": datetime(2026, 10, 2, tzinfo=timezone.utc)}
    with Stubber(client) as stub:
        if case in {"eventually_absent", "timeout"}:
            stub.add_response("describe_secret", deleting, request)
        if case in {"already_absent", "eventually_absent"}:
            stub.add_client_error("describe_secret", service_error_code="ResourceNotFoundException",
                                  expected_params=request)
            wait_for_secret_absence(client, arn, timeout=5)
            assert sleeps == ([5] if case == "eventually_absent" else [])
        elif case == "timeout":
            stub.add_response("describe_secret", deleting, request)
            with pytest.raises(TimeoutError, match="not yet complete"):
                wait_for_secret_absence(client, arn, timeout=5)
            assert sleeps == [5]
        elif case == "not_deleted":
            stub.add_response("describe_secret", {"ARN": arn}, request)
            with pytest.raises(ValueError, match="not marked for deletion"):
                wait_for_secret_absence(client, arn, timeout=5)
            assert not sleeps
        else:
            stub.add_client_error("describe_secret", service_error_code="AccessDeniedException",
                                  expected_params=request)
            with pytest.raises(ClientError, match="AccessDeniedException"):
                wait_for_secret_absence(client, arn, timeout=5)
            assert not sleeps
        stub.assert_no_pending_responses()
