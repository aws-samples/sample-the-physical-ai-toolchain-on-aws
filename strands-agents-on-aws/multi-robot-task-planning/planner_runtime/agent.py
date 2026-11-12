"""The fleet planner on Amazon Bedrock AgentCore Runtime: a Strands Agents agent whose tools talk to the robots over
AWS IoT Core."""

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from planner import plan

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload, context=None):
    return plan(payload["prompt"], min_devices=int(payload.get("min_devices", 1)))  # a new IoT client per request


if __name__ == "__main__":
    app.run()
