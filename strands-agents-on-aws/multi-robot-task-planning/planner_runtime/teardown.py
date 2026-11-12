"""Delete what deploy.py created: the runtime fleet_planner, the role fleet-planner-runtime and the ECR repository
fleet-planner. Then stop the ROS 2 arm container yourself: docker rm -f ros-arm"""

import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-west-2")
s = boto3.Session(region_name=REGION)
ctl, iam = s.client("bedrock-agentcore-control"), s.client("iam")
for r in [r for p in ctl.get_paginator("list_agent_runtimes").paginate() for r in p["agentRuntimes"]]:
    if r["agentRuntimeName"] == "fleet_planner":
        ctl.delete_agent_runtime(agentRuntimeId=r["agentRuntimeId"])
        print("deleted runtime", r["agentRuntimeId"])
try:
    for p in iam.list_role_policies(RoleName="fleet-planner-runtime")["PolicyNames"]:
        iam.delete_role_policy(RoleName="fleet-planner-runtime", PolicyName=p)
    iam.delete_role(RoleName="fleet-planner-runtime")
    print("deleted role")
except iam.exceptions.NoSuchEntityException:
    pass
try:
    s.client("ecr").delete_repository(repositoryName="fleet-planner", force=True)
    print("deleted repository")
except s.client("ecr").exceptions.RepositoryNotFoundException:
    pass
