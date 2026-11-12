"""Create (or update) the fleet planner on Amazon Bedrock AgentCore Runtime: ECR repository, IAM role, runtime
fleet_planner.

The role's IoT grant is the planner's whole reach: connect as planner-*, publish rpcs and its heartbeat, read
presence and its own replies. Publishing a `resume` rpc or an approval answer is denied in IAM, so the planner's
credentials cannot resume
a robot even if its code tried (the robot also demands an operator signature). Teardown: teardown.py.
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import time

import boto3

REGION, REPO, ROLE, RUNTIME, TAG = (
    os.environ.get("AWS_REGION", "us-west-2"),
    "fleet-planner",
    "fleet-planner-runtime",
    "fleet_planner",
    "1",
)
TENANT = "site"
TAGS = {"project": "multi-robot-task-planning-sample"}
s = boto3.Session(region_name=REGION)
BUILDER = os.environ.get("CONTAINER_BUILDER", "finch")  # or docker
acct = s.client("sts").get_caller_identity()["Account"]
ecr, iam, ctl = s.client("ecr"), s.client("iam"), s.client("bedrock-agentcore-control")

try:
    repo = ecr.create_repository(
        repositoryName=REPO,
        imageScanningConfiguration={"scanOnPush": True},
        tags=[{"Key": k, "Value": v} for k, v in TAGS.items()],
    )["repository"]
except ecr.exceptions.RepositoryAlreadyExistsException:
    repo = ecr.describe_repositories(repositoryNames=[REPO])["repositories"][0]
uri = f"{repo['repositoryUri']}:{TAG}"
# The registry password through boto3, so the AWS CLI is not a prerequisite: the token is base64("AWS:<password>").
pw = (
    base64.b64decode(ecr.get_authorization_token()["authorizationData"][0]["authorizationToken"])
    .decode()
    .partition(":")[2]
)
subprocess.run(
    [BUILDER, "login", "--username", "AWS", "--password-stdin", uri.split("/")[0]],
    input=pw,
    text=True,
    check=True,
    capture_output=True,
)
HERE = os.path.dirname(os.path.abspath(__file__))
for f in ("iotlink.py", "planner.py"):  # the planner image carries the same two files the notebook uses
    shutil.copy(os.path.join(HERE, "..", "fleet", f), HERE)
subprocess.run([BUILDER, "build", "--platform", "linux/arm64", "-t", f"fleet-planner:{TAG}", HERE], check=True)
subprocess.run([BUILDER, "tag", f"fleet-planner:{TAG}", uri], check=True)
subprocess.run([BUILDER, "push", uri], check=True, capture_output=True)
print("pushed", uri)

trust = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": acct},
                "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock-agentcore:{REGION}:{acct}:*"},
            },
        }
    ],
}
try:
    role_arn = iam.create_role(
        RoleName=ROLE,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description="Fleet planner on Amazon Bedrock AgentCore Runtime",
        Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()],
    )["Role"]["Arn"]
    new = True
except iam.exceptions.EntityAlreadyExistsException:
    role_arn, new = iam.get_role(RoleName=ROLE)["Role"]["Arn"], False
iot = f"arn:aws:iot:{REGION}:{acct}"
policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "Model",
            "Effect": "Allow",
            "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/anthropic.*",
                f"arn:aws:bedrock:*:{acct}:inference-profile/*",
            ],
        },
        {
            "Sid": "Image",
            "Effect": "Allow",
            "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
            "Resource": repo["repositoryArn"],
        },
        {"Sid": "ImageAuth", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
        {
            "Sid": "Logs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams",
                "logs:DescribeLogGroups",
            ],
            "Resource": f"arn:aws:logs:{REGION}:{acct}:log-group:/aws/bedrock-agentcore/runtimes/*",
        },
        {"Sid": "IotConnect", "Effect": "Allow", "Action": "iot:Connect", "Resource": f"{iot}:client/planner-*"},
        {
            "Sid": "IotPublish",
            "Effect": "Allow",
            "Action": "iot:Publish",
            "Resource": [f"{iot}:topic/{TENANT}/*/rpc/*", f"{iot}:topic/{TENANT}/heartbeat/${{iot:ClientId}}"],
        },
        {
            "Sid": "NoResumeNoApproval",
            "Effect": "Deny",
            "Action": "iot:Publish",
            "Resource": [f"{iot}:topic/{TENANT}/*/rpc/resume", f"{iot}:topic/{TENANT}/*/approval"],
        },
        {
            "Sid": "IotSubscribe",
            "Effect": "Allow",
            "Action": "iot:Subscribe",
            "Resource": [
                f"{iot}:topicfilter/{TENANT}/presence/+",
                f"{iot}:topicfilter/{TENANT}/reply/${{iot:ClientId}}",
                f"{iot}:topicfilter/$aws/events/presence/disconnected/+",
            ],
        },
        {
            "Sid": "IotReceive",
            "Effect": "Allow",
            "Action": "iot:Receive",
            "Resource": [
                f"{iot}:topic/{TENANT}/presence/*",
                f"{iot}:topic/{TENANT}/reply/${{iot:ClientId}}",
                f"{iot}:topic/$aws/events/presence/disconnected/*",
            ],
        },
        # each robot's description is a retained message; a wildcard subscription doesn't deliver those, so read them
        {"Sid": "RetainedList", "Effect": "Allow", "Action": "iot:ListRetainedMessages", "Resource": "*"},
        {
            "Sid": "RetainedGet",
            "Effect": "Allow",
            "Action": "iot:GetRetainedMessage",
            "Resource": f"{iot}:topic/{TENANT}/presence/*",
        },
    ],
}
iam.put_role_policy(RoleName=ROLE, PolicyName="fleet-planner", PolicyDocument=json.dumps(policy))
print("role", role_arn, "(new)" if new else "(existing)")
if new:
    time.sleep(15)

existing = [
    r
    for p in ctl.get_paginator("list_agent_runtimes").paginate()
    for r in p["agentRuntimes"]
    if r["agentRuntimeName"] == RUNTIME
]
args = dict(
    agentRuntimeArtifact={"containerConfiguration": {"containerUri": uri}},
    roleArn=role_arn,
    networkConfiguration={"networkMode": "PUBLIC"},
    environmentVariables={
        "AWS_REGION": REGION,
        "IOT_ENDPOINT": s.client("iot").describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"],
    },
)
if existing:
    rt = ctl.update_agent_runtime(agentRuntimeId=existing[0]["agentRuntimeId"], **args)
else:
    rt = ctl.create_agent_runtime(agentRuntimeName=RUNTIME, description="Fleet planner", tags=TAGS, **args)
t0 = time.time()
while (d := ctl.get_agent_runtime(agentRuntimeId=rt["agentRuntimeId"]))["status"] not in (
    "READY",
    "CREATE_FAILED",
    "UPDATE_FAILED",
):
    time.sleep(10)
print(f"runtime {d['agentRuntimeArn']} status={d['status']} after {time.time() - t0:.0f}s")
if d["status"] != "READY":
    print(json.dumps({k: d.get(k) for k in ("failureReason", "statusReasons")}, default=str))
    sys.exit(1)
open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime_arn.txt"), "w").write(
    d["agentRuntimeArn"]
)
