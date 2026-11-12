FROM --platform=linux/arm64 public.ecr.aws/docker/library/python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir bedrock-agentcore "strands-agents>=1.43" "awsiotsdk>=1.31"
COPY agent.py iotlink.py planner.py ./
EXPOSE 8080
CMD ["python", "agent.py"]
