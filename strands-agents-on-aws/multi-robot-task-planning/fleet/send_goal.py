"""Send one goal to the planner on Amazon Bedrock AgentCore Runtime and print its report.

  python send_goal.py "Start the morning shift. ..."

A long goal must never be sent twice, so the client waits up to 15 minutes and does not retry.
"""

import json
import sys
import uuid
from pathlib import Path

import boto3
from botocore.config import Config

arn = (Path(__file__).resolve().parent.parent / "runtime_arn.txt").read_text().strip()
agentcore = boto3.client(
    "bedrock-agentcore",
    region_name=arn.split(":")[3],
    config=Config(read_timeout=900, retries={"total_max_attempts": 1}),
)  # 1 = the first attempt only
r = agentcore.invoke_agent_runtime(
    agentRuntimeArn=arn,
    runtimeSessionId=f"goal-{uuid.uuid4().hex}",
    payload=json.dumps({"prompt": sys.argv[1]}).encode(),
)
out = json.loads(r["response"].read())
print(out.get("reply", out))
