"""A minimal request/reply link over AWS IoT Core (MQTT5, AWS IoT Device SDK v2), shared by robot hosts and the planner.

Topics, all under a tenant prefix: <tenant>/presence/<device> (heartbeat), <tenant>/<device>/rpc/<name> (requests),
<tenant>/reply/<client> (replies to one caller). The test signs with SigV4 (IAM); production robots use one X.509
certificate per thing (mqtt5_client_builder.mtls_from_path) with an IoT policy scoped to their own topics.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import Future
from typing import Any, Callable

from awscrt import auth, mqtt5
from awsiot import mqtt5_client_builder


def _region() -> str:
    """The sample's one Region rule, the same as deploy.py, launch.py and the notebook: AWS_REGION, else us-west-2."""
    return os.environ.get("AWS_REGION", "us-west-2")


def _endpoint(region: str) -> str:
    """Your account's AWS IoT Core data endpoint in `region`: IOT_ENDPOINT if set, else looked up once with
    DescribeEndpoint. The Region is passed explicitly because boto3 does not read AWS_REGION."""
    if os.environ.get("IOT_ENDPOINT"):
        return os.environ["IOT_ENDPOINT"]
    import boto3

    return boto3.client("iot", region_name=region).describe_endpoint(endpointType="iot:Data-ATS")["endpointAddress"]


class Link:
    def __init__(
        self,
        client_id: str,
        tenant: str = "site",
        region: str | None = None,
        endpoint: str | None = None,
        will: str | None = None,
    ):
        """`will`: a topic whose retained message IoT Core clears (a retained empty payload) if this connection drops
        without a clean disconnect. IoT Core sends no Will on a clean disconnect, so the owner clears it itself."""
        self.client_id, self.tenant, self._subs, self._waiting = client_id, tenant, {}, {}
        self.on_reconnect: list[Callable[[], None]] = []
        region = region or _region()  # the endpoint and the SigV4 signature must name the same Region
        self.region, self.endpoint = region, endpoint or _endpoint(region)
        up = Future()
        connect = (
            mqtt5.ConnectPacket(
                will=mqtt5.PublishPacket(topic=will, payload=b"", qos=mqtt5.QoS.AT_LEAST_ONCE, retain=True)
            )
            if will
            else None
        )
        self.client = mqtt5_client_builder.websockets_with_default_aws_signing(
            endpoint=self.endpoint,
            region=region,
            credentials_provider=auth.AwsCredentialsProvider.new_default_chain(),
            client_id=client_id,
            clean_start=True,
            keep_alive_secs=30,
            connect_options=connect,
            on_publish_received=self._on_message,
            on_lifecycle_connection_success=lambda _e: self._connected(up),
            on_lifecycle_connection_failure=lambda e: up.done() or up.set_exception(RuntimeError(str(e.exception))),
        )
        self.client.start()
        up.result(timeout=20)
        self.subscribe(f"{tenant}/reply/{client_id}", self._on_reply)

    def _connected(self, up: Future) -> None:
        """Every successful connection. clean_start=True means a reconnect starts a new session with NO
        subscriptions: without this, a robot that reconnects after a network blip keeps publishing presence (it
        looks online) but never hears another command."""
        if up.done():
            for topic in list(self._subs):
                self.client.subscribe(mqtt5.SubscribePacket([mqtt5.Subscription(topic, mqtt5.QoS.AT_LEAST_ONCE)]))
            self.reconnects = getattr(self, "reconnects", 0) + 1
            for hook in list(self.on_reconnect):  # e.g. a robot republishes its retained description
                threading.Thread(target=hook, daemon=True).start()
        else:
            up.set_result(True)

    def subscribe(self, topic: str, handler: Callable[[str, dict], None]) -> None:
        """Add a handler; several handlers may share one topic filter (a second call no longer replaces the first)."""
        first = topic not in self._subs
        self._subs.setdefault(topic, []).append(handler)
        if first:
            self.client.subscribe(mqtt5.SubscribePacket([mqtt5.Subscription(topic, mqtt5.QoS.AT_LEAST_ONCE)])).result(
                10
            )

    def publish(self, topic: str, body: dict | None, retain: bool = False) -> None:
        """`body=None` sends an empty payload: with `retain=True` that deletes the topic's retained message."""
        payload = b"" if body is None else json.dumps(body).encode()
        self.client.publish(
            mqtt5.PublishPacket(topic=topic, payload=payload, qos=mqtt5.QoS.AT_LEAST_ONCE, retain=retain)
        )

    def _on_message(self, data: Any) -> None:
        topic = data.publish_packet.topic
        raw = bytes(data.publish_packet.payload or b"")
        try:
            body = json.loads(raw) if raw else {}  # an empty payload (a cleared retained message) arrives as {}
        except ValueError:
            return
        for pattern, handlers in list(self._subs.items()):
            if _match(pattern, topic):
                for handler in list(handlers):
                    threading.Thread(target=handler, args=(topic, body), daemon=True).start()

    def _on_reply(self, _topic: str, body: dict) -> None:
        fut = self._waiting.pop(body.get("id"), None)
        if fut is not None and not fut.done():
            fut.set_result(body)

    def call(self, device: str, rpc: str, timeout: float = 6.0, **args: Any) -> dict:
        """One request, one reply. No reply means the device is offline or unreachable: nothing is queued or retried."""
        rid = uuid.uuid4().hex
        fut = self._waiting[rid] = Future()
        self.publish(
            f"{self.tenant}/{device}/rpc/{rpc}", {"id": rid, "from": self.client_id, "args": args, "ts": time.time()}
        )
        try:
            body = fut.result(timeout=timeout)
        except TimeoutError:
            self._waiting.pop(rid, None)
            raise ConnectionError(
                f"{device} did not answer {rpc} within {timeout:g} s (offline or unreachable); nothing was queued"
            )
        if "error" in body:
            raise RuntimeError(f"{device} refused {rpc}: {body['error']}")
        return body["result"]

    def close(self) -> None:
        self.client.stop()


def _match(pattern: str, topic: str) -> bool:
    p, t = pattern.split("/"), topic.split("/")
    return len(p) == len(t) and all(a in ("+", b) for a, b in zip(p, t))
