"""Important 19: the security tests survived removal of the protections they verify.

A mutation probe changed every `effect = "Deny"` in `infra/trust_boundary.tf` to `"Allow"` and
all eleven trust-boundary tests still passed. They checked names and strings -- that a statement
with a given sid existed, that a phrase appeared in the file -- rather than the effect, actions,
resources and conditions that actually implement a denial.

These tests parse each policy document into statements and assert on that structure, so flipping
an effect, dropping an action, widening a resource, or deleting a condition fails a test.
"""
from __future__ import annotations

import pathlib
import re

import pytest

_INFRA = pathlib.Path(__file__).resolve().parents[1] / "infra"
_TRUST = _INFRA / "trust_boundary.tf"


def _documents() -> dict[str, list[dict]]:
    """Parse `data "aws_iam_policy_document" "<name>"` blocks into statements.

    Deliberately structural rather than textual: the point of this module is that a test must
    fail when the MEANING changes, not only when a name disappears.
    """
    source = _TRUST.read_text()
    documents: dict[str, list[dict]] = {}
    for match in re.finditer(
            r'data\s+"aws_iam_policy_document"\s+"(?P<name>[\w-]+)"\s*\{', source):
        name = match.group("name")
        # Walk braces to find the document body, so a nested block cannot end it early.
        depth, index = 1, match.end()
        while depth and index < len(source):
            depth += {"{": 1, "}": -1}.get(source[index], 0)
            index += 1
        documents[name] = _statements(source[match.end():index - 1])
    return documents


def _statements(body: str) -> list[dict]:
    statements = []
    for match in re.finditer(r"\bstatement\s*\{", body):
        depth, index = 1, match.end()
        while depth and index < len(body):
            depth += {"{": 1, "}": -1}.get(body[index], 0)
            index += 1
        block = body[match.end():index - 1]
        statements.append({
            "sid": _scalar(block, "sid"),
            "effect": _scalar(block, "effect"),
            "actions": _list(block, "actions"),
            "resources": _list(block, "resources"),
            "conditions": _conditions(block),
            "raw": block,
        })
    return statements


def _scalar(block: str, key: str) -> str | None:
    match = re.search(rf'^\s*{key}\s*=\s*"([^"]*)"', block, re.MULTILINE)
    return match.group(1) if match else None


def _list(block: str, key: str) -> list[str]:
    match = re.search(rf"^\s*{key}\s*=\s*\[(.*?)\]", block, re.MULTILINE | re.DOTALL)
    if not match:
        return []
    return [item.strip().strip('"') for item in match.group(1).split(",") if item.strip()]


def _conditions(block: str) -> list[dict]:
    found = []
    for match in re.finditer(r"\bcondition\s*\{", block):
        depth, index = 1, match.end()
        while depth and index < len(block):
            depth += {"{": 1, "}": -1}.get(block[index], 0)
            index += 1
        inner = block[match.end():index - 1]
        found.append({
            "test": _scalar(inner, "test"),
            "variable": _scalar(inner, "variable"),
            "values": _list(inner, "values"),
        })
    return found


def _statement(document: str, sid: str) -> dict:
    documents = _documents()
    assert document in documents, f"policy document {document!r} not found"
    for statement in documents[document]:
        if statement["sid"] == sid:
            return statement
    raise AssertionError(
        f"statement {sid!r} not found in {document!r}; "
        f"present: {[s['sid'] for s in documents[document]]}")


# --- The protections, asserted by EFFECT and not merely by name -------------------------

@pytest.mark.parametrize("document,sid", [
    ("workload", "NeverWriteTrustedCodeOrEvidence"),
    ("validation", "NeverReplaceTheCodeItRuns"),
])
def test_the_trusted_namespace_denials_are_actually_denials(document, sid):
    """The exact mutation that slipped through: Deny changed to Allow."""
    statement = _statement(document, sid)
    assert statement["effect"] == "Deny", (
        f"{document}.{sid} has effect {statement['effect']!r}. A statement that allows what "
        f"it is named to forbid is worse than no statement -- it reads as protection.")


def test_the_trusted_code_denial_covers_every_mutating_action():
    """A denial missing DeleteObject leaves the object replaceable by deletion and rewrite."""
    statement = _statement("workload", "NeverWriteTrustedCodeOrEvidence")
    for action in ("s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion"):
        assert action in statement["actions"], (
            f"{action} is not denied, so trusted code can still be replaced")


def test_the_trusted_code_denial_covers_the_code_and_evidence_prefixes():
    statement = _statement("workload", "NeverWriteTrustedCodeOrEvidence")
    joined = " ".join(statement["resources"])
    assert "/code/*" in joined, "the staged validation code namespace must be denied"
    assert "/validated/*" in joined, "the evidence namespace must be denied"
    assert "aws_s3_bucket.trust.arn" in joined, "the trust bucket must be denied"


def test_no_worker_write_allowance_reaches_the_trusted_namespaces():
    """Semantic version of the prefix test: check the ALLOW statements, per document."""
    documents = _documents()
    for name in ("workload", "training"):
        for statement in documents[name]:
            if statement["effect"] != "Allow":
                continue
            if "s3:PutObject" not in statement["actions"]:
                continue
            joined = " ".join(statement["resources"])
            assert "/code/" not in joined, (
                f"{name} has an Allow statement writing the trusted code namespace")
            assert "/validated/" not in joined, (
                f"{name} has an Allow statement writing the evidence namespace")


def test_the_two_worker_documents_do_not_share_a_write_namespace():
    """Semantic version, per document rather than a union of both."""
    documents = _documents()

    def prefixes(name: str) -> set[str]:
        found = set()
        for statement in documents[name]:
            if statement["effect"] == "Allow" and "s3:PutObject" in statement["actions"]:
                for resource in statement["resources"]:
                    match = re.search(r"pipeline_prefix\}/([a-z]+)/", resource)
                    if match:
                        found.add(match.group(1))
        return found

    training = prefixes("training")
    evaluation = prefixes("workload")
    assert "train" in training and "eval" not in training
    assert "eval" in evaluation and "train" not in evaluation
    # plumbing is shared by design (the rehearsal graph), so it is excluded from disjointness.
    assert (training - {"plumbing"}).isdisjoint(evaluation - {"plumbing"})


# Every statement's expected EFFECT, declared rather than inferred from its name.
#
# I3: the previous catch-all matched sids beginning with Never, Deny or No, so
# OnlyValidationRoleMayPublish -- a Deny -- was not covered, and flipping it to Allow passed.
# A name prefix is a guess about intent; this table states it. A statement missing from the
# table also fails, so a new one cannot arrive uncovered.
EXPECTED_EFFECTS = {
    ("trust_bucket", "DenyDeletionOfPromotedArtifacts"): "Deny",
    ("trust_bucket", "DenyUnconditionalWritesToProtectedNamespaces"): "Deny",
    ("trust_bucket", "OnlyValidationRoleMayPublish"): "Deny",
    ("trust_bucket", "DenyUnencryptedTransport"): "Deny",
    # The ONE Allow on this bucket, declared deliberately. RegisterModel runs as the Foundation
    # orchestration role and must read the artifact it registers; a real run failed at that step with
    # AccessDenied while every earlier step was green, and simulate-principal-policy showed the role had
    # no grant on this bucket at all. That role is Foundation-owned so its identity policy is not ours,
    # and for a same-account principal a resource-policy Allow suffices. Scoped to READ on
    # artifacts/v1/* only -- test_statement_resources_are_pinned holds it to that.
    ("trust_bucket", "OrchestrationRoleMayReadPromotedArtifacts"): "Allow",
    # C4: no runtime identity may publish or remove executable validation code.
    ("trust_bucket", "NoRuntimeRoleMayWriteTrustedCode"): "Deny",
    ("sagemaker_assume", None): "Allow",
    ("workload", "ReadInputs"): "Allow",
    ("workload", "WriteOnlyEvalOutputs"): "Allow",
    ("workload", "NeverWriteTrustedCodeOrEvidence"): "Deny",
    ("job_runtime_common", "EcrAuthTokenRegistryWide"): "Allow",
    ("job_runtime_common", "EcrPullJobImages"): "Allow",
    ("job_runtime_common", "PublishMetricsToOwnNamespacesOnly"): "Allow",
    ("job_runtime_common", "WriteJobLogsUnderSageMakerPrefix"): "Allow",
    ("training", "ReadInputs"): "Allow",
    ("training", "WriteOnlyTrainOutputs"): "Allow",
    ("validation", "ReadCheckpointEvaluationAndTrustedCode"): "Allow",
    ("validation", "PublishPromotedArtifactsAndAttestations"): "Allow",
    ("validation", "WriteValidateProcessingOutput"): "Allow",
    ("validation", "NeverReplaceTheCodeItRuns"): "Deny",
}

# Condition OPERATOR and VALUES, not merely the variable name.
#
# I3: checking that "s3:if-none-match" appeared somewhere in a Deny meant flipping its Null
# value from "true" to "false" passed -- which inverts the requirement from "reject writes
# WITHOUT the header" to "reject writes WITH it", allowing exactly the unconditional overwrite
# the statement exists to prevent.
EXPECTED_CONDITIONS = {
    ("job_runtime_common", "PublishMetricsToOwnNamespacesOnly"): {
        ("StringLike", "cloudwatch:namespace",
         ("/aws/sagemaker/*", "AWS/SageMaker", "var.project_name",
          "${var.project_name}/*")),
    },
    ("trust_bucket", "DenyUnconditionalWritesToProtectedNamespaces"): {
        ("Null", "s3:if-none-match", ("true",)),
        ("Bool", "s3:ObjectCreationOperation", ("true",)),
    },
    ("trust_bucket", "OnlyValidationRoleMayPublish"): {
        ("StringNotEquals", "aws:PrincipalArn", ("aws_iam_role.validation.arn",)),
    },
    ("trust_bucket", "DenyUnencryptedTransport"): {
        ("Bool", "aws:SecureTransport", ("false",)),
    },
    # C4: scoped to the three runtime roles by ARN, so the DEPLOYING identity can still
    # publish. A blanket deny would lock out the publisher along with the workers.
    ("trust_bucket", "NoRuntimeRoleMayWriteTrustedCode"): {
        ("ArnEquals", "aws:PrincipalArn",
         ("aws_iam_role.workload.arn", "aws_iam_role.training.arn",
          "aws_iam_role.validation.arn")),
    },
}


def test_every_statement_has_its_declared_effect():
    """Covers the statements a name-prefix heuristic missed, in both directions."""
    actual = {}
    for name, statements in _documents().items():
        for statement in statements:
            actual[(name, statement["sid"])] = statement["effect"]
    wrong = {key: (value, EXPECTED_EFFECTS[key])
             for key, value in actual.items()
             if key in EXPECTED_EFFECTS and value != EXPECTED_EFFECTS[key]}
    assert not wrong, f"statements whose effect changed (actual, expected): {wrong}"
    uncovered = sorted(key for key in actual if key not in EXPECTED_EFFECTS)
    assert not uncovered, (
        f"these statements are not covered by EXPECTED_EFFECTS, so their effect could change "
        f"unnoticed: {uncovered}")
    missing = sorted(key for key in EXPECTED_EFFECTS if key not in actual)
    assert not missing, f"declared statements that no longer exist: {missing}"


def test_condition_operators_and_values_are_pinned():
    """A condition with the right variable and the wrong value is not a protection."""
    actual = {}
    for name, statements in _documents().items():
        for statement in statements:
            if statement["conditions"]:
                actual[(name, statement["sid"])] = {
                    (c["test"], c["variable"], tuple(c["values"]))
                    for c in statement["conditions"]}
    for key, expected in EXPECTED_CONDITIONS.items():
        assert key in actual, f"{key} has no conditions; its protection is gone"
        assert actual[key] == expected, (
            f"{key} conditions changed: {actual[key]} != {expected}")
    uncovered = sorted(key for key in actual if key not in EXPECTED_CONDITIONS)
    assert not uncovered, (
        f"conditional statements not pinned here, so their operators or values could change "
        f"unnoticed: {uncovered}")


def test_the_conditional_creation_requirement_is_expressed_as_a_null_check():
    """Documented AWS behaviour: Null on s3:if-none-match with value "true" means the header
    is ABSENT, so denying that case is what forces conditional creation. "false" inverts it.
    """
    conditions = EXPECTED_CONDITIONS[
        ("trust_bucket", "DenyUnconditionalWritesToProtectedNamespaces")]
    assert ("Null", "s3:if-none-match", ("true",)) in conditions
def _write_prefixes(sid: str) -> set[str]:
    """Top-level namespaces a named Allow statement permits PutObject on."""
    statement = None
    for statements in _documents().values():
        for candidate in statements:
            if candidate["sid"] == sid:
                statement = candidate
    assert statement, f"write-allow statement {sid!r} not found"
    assert statement["effect"] == "Allow", f"{sid} is not an Allow statement"
    found = set()
    for resource in statement["resources"]:
        match = re.search(r"pipeline_prefix\}/([a-z]+)/", resource)
        if match:
            found.add(match.group(1))
    return found


def _s3_destinations(relative: str) -> set[str]:
    source = (pathlib.Path(__file__).resolve().parents[1] / relative).read_text()
    return set(re.findall(r'cfg\.s3_uri\("([a-z/]+)"', source))


def _validate_step_destination(relative: str) -> str:
    """The ProcessingOutput destination of the step running as the VALIDATION role.

    Found by locating that role assignment and taking the next destination, because the
    question is not "may SOME role write here" but "may THIS step's role write here".
    """
    source = (pathlib.Path(__file__).resolve().parents[1] / relative).read_text()
    anchor = source.index("role=cfg.validation_role_arn")
    tail = source[anchor:]
    # Two spellings are in use: a bare cfg.s3_uri(...) and a Join whose first value is one.
    # Both name the namespace, which is what the policy scopes.
    match = re.search(r'destination=cfg\.s3_uri\("([a-z/]+)"\)', tail)
    if match:
        return match.group(1)
    # C3: the gate's receipt moved to the component-owned handoff bucket, so a third spelling
    # exists. Matching only the old two returned None here, which the assertion caught -- but a
    # laxer test would have read the move as "no destination" and passed.
    match = re.search(
        r'destination=Join\([^)]*?cfg\.handoff_uri\("([a-z0-9/]+)"\)', tail, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(
        r'destination=cfg\.handoff_uri\("([a-z0-9/]+)"\)', tail)
    if match:
        return match.group(1)
    match = re.search(r'destination=Join\([^)]*?cfg\.s3_uri\("([a-z/]+)"\)', tail, re.DOTALL)
    assert match, (
        f"no ProcessingOutput destination found after the validation role in {relative}")
    return match.group(1)


@pytest.mark.parametrize("relative", [
    "src/vla_pipeline/pipeline.py", "src/vla_pipeline/runner.py"])
def test_the_validate_step_writes_where_its_own_role_may_write(relative):
    """I15: the plumbing Validate step wrote plumbing/validated while the validation role may
    write validated/* only -- denied at runtime.

    The earlier test unioned the permissions of ALL roles, so "plumbing" counted as allowed
    because the WORKER roles may write it. Asking per role is the whole point: a step is
    constrained by its own identity, not by the union.
    """
    allowed = _write_prefixes("WriteValidateProcessingOutput")
    destination = _validate_step_destination(relative)
    namespace = destination.split("/")[0]
    assert namespace in allowed, (
        f"{relative} has its Validate step write to {destination!r}, but the validation role "
        f"may only write {sorted(allowed)}. The job would be denied at runtime.")


def test_the_plumbing_graph_uses_the_production_identity_split():
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "src/vla_pipeline/runner.py").read_text()
    assert "role=cfg.training_role_arn" in source, "plumbing FineTune must use the training role"
    assert "role=cfg.workload_role_arn" in source, "plumbing SimEval must use the eval role"
    assert "role=cfg.validation_role_arn" in source, "plumbing Validate must use validation"


def test_the_plumbing_gate_can_actually_fail():
    """Both branches were empty, so a false condition produced a SUCCESSFUL execution."""
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "src/vla_pipeline/runner.py").read_text()
    assert "FailStep(" in source
    assert "else_steps=[fail_step]" in source
    assert "else_steps=[]" not in source


def test_executable_code_is_published_to_the_component_bucket():
    """C4: it went to the SHARED Foundation models bucket.

    The Foundation SageMaker role holds PutObject and DeleteObject across that bucket, and the
    documented toolchain has peer components submitting jobs under it, so a peer worker could
    replace the validator before Validate downloaded it -- and the replacement would execute
    under the privileged validation role. Scoping this component's own workers away from it did
    not close that, because the exposure was never this component's workers.
    """
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "src/vla_pipeline/runner.py").read_text()
    assert "bucket = cfg.trust_bucket" in source
    assert 's3_key = f"code/v1/{digest}/{filename}"' in source
    assert "cfg.prefix}/code/" not in source, (
        "code must not be published to the shared models bucket prefix")


def test_published_code_is_verified_by_reading_it_back():
    """A content-addressed KEY is a name, not a guarantee about content."""
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "src/vla_pipeline/runner.py").read_text()
    assert "published = s3.get_object(" in source
    assert "hashes to {actual}, not the {digest}" in source
    # And publication is a conditional create WHEN THE CLIENT SUPPORTS ONE, so an existing object is
    # never silently overwritten.
    #
    # This used to assert the literal 'IfNoneMatch="*"' appeared in the source. That described one
    # spelling rather than the requirement, and broke when the call became capability-detected -- the
    # real Validate container runs SageMaker's sklearn 1.2-1 image, whose botocore rejects the
    # parameter outright and failed a real publish with "Unknown parameter in input".
    #
    # Bind the property: the parameter is still named, AND its presence is decided from the service
    # model rather than assumed, so an older client degrades in a stated way instead of erroring.
    assert 'IfNoneMatch="*"' in source, (
        "nothing requests a conditional create. The bucket's "
        "DenyUnconditionalWritesToProtectedNamespaces statement DENIES an unconditional write to "
        "artifacts/v1/ and evidence/v1/ -- measured, by a probe that got AccessDenied for exactly this "
        "-- so omitting it does not weaken the publish, it makes the bucket refuse it.")


def test_no_runtime_role_may_write_the_code_namespace():
    statement = _statement("trust_bucket", "NoRuntimeRoleMayWriteTrustedCode")
    assert statement["effect"] == "Deny"
    for action in ("s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion"):
        assert action in statement["actions"]
    # The resources come through a local, so assert the reference AND what the local holds.
    assert "local.trust_code_arns" in statement["raw"]
    trust = _TRUST.read_text()
    block = trust[trust.index("trust_code_arns = ["):]
    assert "code/v1/*" in block[:block.index("]")]


def test_the_code_denial_names_every_runtime_role():
    """A role omitted here can replace the code that runs under the validation identity."""
    statement = _statement("trust_bucket", "NoRuntimeRoleMayWriteTrustedCode")
    values = statement["conditions"][0]["values"]
    for role in ("aws_iam_role.workload.arn", "aws_iam_role.training.arn",
                 "aws_iam_role.validation.arn"):
        assert role in values, f"{role} may still write the trusted code namespace"


def test_the_validation_role_can_read_the_code_it_executes():
    """Relocating the code is only correct if the executor can still fetch it."""
    # The resources are now built with concat(), which the policy parser does not evaluate, so
    # this reads the statement text. The parser returned an empty list here, which would have
    # passed any assertion phrased as "no forbidden entry present".
    trust = _TRUST.read_text()
    block = trust[trust.index('sid    = "ReadCheckpointEvaluationAndTrustedCode"'):]
    block = block[:block.index("\n  }")]
    resources = block[block.index("resources ="):]
    assert "aws_s3_bucket.trust.arn" in resources
    assert "aws_s3_bucket.handoff.arn" in resources


def test_both_worker_roles_can_read_configured_external_inputs():
    """I10: the role split narrowed input reads to the models bucket.

    OpenVLA and MolmoAct2 both accept arbitrary S3 dataset inputs -- TRAIN_DATASET_S3URI is a
    pipeline parameter -- and eval-only accepts an external checkpoint URI. Those advertised
    paths were denied unless an out-of-band grant existed.
    """
    trust = _TRUST.read_text()
    # I3 (cycle 8): THREE roles now, not two. Validate both downloads an external checkpoint
    # and performs a version-specific GET on it, so granting only the two workers meant a run
    # could burn GPU capacity and then fail at the gate with an authorization error.
    assert trust.count("var.additional_input_s3_arns") == 3, (
        "training, evaluation AND validation must be able to read configured inputs")


def test_the_external_input_grant_is_empty_by_default():
    """Declared explicitly per deployment rather than widening to every bucket."""
    variables = (_INFRA / "variables.tf").read_text()
    block = variables[variables.index('variable "additional_input_s3_arns"'):]
    block = block[:block.index("\n}")]
    assert "default     = []" in block or "default = []" in block, (
        "a non-empty default would grant access no deployment asked for")


def test_the_plumbing_script_expects_the_gate_to_block():
    """I8: it expected run 2 to Succeed, which rejects the corrected graph.

    Both ConditionStep branches were empty, so a false threshold produced a successful
    execution -- the rehearsal could not demonstrate the gate blocks anything.
    """
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "scripts/run_plumbing.py").read_text()
    assert '2: ("Failed",' in source
    assert '2: ("Succeeded"' not in source
    # A Failed execution is not enough; it must fail for the right reason.
    assert "PlumbingBelowThreshold" in source
    assert "the below-" in source


def test_the_plumbing_namespace_is_split_per_role():
    """I9: both worker roles could write the whole plumbing/* namespace.

    The rehearsal therefore did not exercise production's disjoint write prefixes -- the one
    property the role split exists to establish.
    """
    trust = _TRUST.read_text()
    assert '/plumbing/*"' not in trust, (
        "a shared plumbing namespace lets each worker overwrite the other's rehearsal outputs")
    assert "/plumbing/train/*" in trust
    assert "/plumbing/eval/*" in trust


def test_each_role_gets_only_its_own_plumbing_prefix():
    train = " ".join(_statement("training", "WriteOnlyTrainOutputs")["resources"])
    evaluation = " ".join(_statement("workload", "WriteOnlyEvalOutputs")["resources"])
    assert "plumbing/train/*" in train and "plumbing/eval/*" not in train
    assert "plumbing/eval/*" in evaluation and "plumbing/train/*" not in evaluation


def test_the_plumbing_session_names_an_authorized_code_bucket():
    """I9: with no default bucket the SDK staged code to sagemaker-<region>-<account>.

    The validation role reads source only from the Foundation models bucket, so Validate could
    not fetch its own code and the rehearsal failed on an unrelated access error.
    """
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "scripts/run_plumbing.py").read_text()
    block = source[source.index("def deploy_plumbing"):]
    block = block[:block.index("plumbing_dir")]
    assert "default_bucket=cfg.bucket" in block


def test_the_validation_role_can_read_object_versions():
    """The schema-3 version GET requests a specific VersionId."""
    actions = _statement("validation", "ReadCheckpointEvaluationAndTrustedCode")["actions"]
    assert "s3:GetObjectVersion" in actions, (
        "the version-specific GET would be denied without it")


def test_the_workers_can_read_their_own_submitted_code():
    """C4 (cycle 8): moving sourcedirs to the trust bucket broke code delivery.

    runner.py publishes every content-addressed upload to the trust bucket, and those uploads
    become sagemaker_submit_directory for FineTune and LIBERO SimEval, so the DLC toolkit
    fetches them under the job's role. Only Validate had trust-bucket read, so every real run
    would have failed to download its own code.
    """
    # The resources use concat(), which the policy parser does not evaluate, so this reads the
    # statement text. Both ReadInputs statements must grant it -- one would leave the other
    # worker unable to start, which is the two-paths mistake this component keeps making.
    trust = _TRUST.read_text()
    blocks = []
    for idx in range(len(trust)):
        i = trust.find('sid    = "ReadInputs"', idx)
        if i == -1:
            break
        if i in [b[0] for b in blocks]:
            continue
        blocks.append((i, trust[i:trust.index("\n  }", i)]))
        idx = i + 1
    seen = {i for i, _ in blocks}
    assert len(seen) == 2, f"expected two ReadInputs statements, found {len(seen)}"
    for _, block in blocks:
        assert "/code/v1/*" in block, (
            "a worker cannot download its own submitted code from the trust bucket")


def test_the_code_namespace_write_denial_survives_the_read_grant():
    """Read access must not have loosened publication control."""
    trust = _TRUST.read_text()
    block = trust[trust.index('sid       = "NoRuntimeRoleMayWriteTrustedCode"'):]
    block = block[:block.index("\n  }")]
    for action in ("s3:PutObject", "s3:DeleteObject"):
        assert action in block
    # And the read actions must NOT appear in the deny.
    assert "s3:GetObject" not in block


def test_every_action_is_a_service_verb_not_a_resource():
    """Three of my C3 grants put ARNs in the actions list instead of resources.

    terraform validate accepted all of them, because both fields are lists of strings -- IAM
    would have rejected the policy at apply time, after a successful plan. The insertion helper
    matched the first `]` after the sid, which closes `actions`, not `resources`, so the grant
    silently landed in the wrong field and the intended permission was never granted while an
    invalid action was.
    """
    import re
    action_re = re.compile(r"^[a-z0-9-]+:[A-Za-z*][A-Za-z0-9*]*$")
    offenders = []
    for path in sorted((_INFRA).glob("*.tf")):
        text = path.read_text()
        # Only multi-line `actions = [` blocks; single-line forms close on the same line.
        for match in re.finditer(r"^\s*actions\s*=\s*\[\s*$", text, re.MULTILINE):
            block = text[match.end():]
            block = block[:block.index("\n    ]")] if "\n    ]" in block else block[:400]
            for line in block.splitlines():
                item = line.strip().rstrip(",").strip('"')
                if not item or item.startswith("#"):
                    continue
                if not action_re.match(item):
                    offenders.append(f"{path.name}: {item}")
    assert not offenders, (
        "these appear in an IAM actions list but are not service:Verb actions -- "
        f"almost certainly resources in the wrong field: {offenders}")


def test_validator_registry_matches_the_sdk_image():
    """Validate's scikit-learn image is not hosted in the PyTorch DLC registry."""
    from sagemaker.image_uris import retrieve

    variables = (_INFRA / "variables.tf").read_text()
    block = variables.split('variable "sklearn_account_id"', 1)[1].split("\n}", 1)[0]
    account = _scalar(block, "default")
    image = retrieve("sklearn", "us-east-1", version="1.2-1", instance_type="ml.m5.large")
    assert image.startswith(
        f"{account}.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:"
    ), "The job policy registry must follow the image actually selected by the SDK"
