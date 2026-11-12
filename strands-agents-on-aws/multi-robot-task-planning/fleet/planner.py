"""The planner's side: Strands Agents @tool functions over AWS IoT Core. There is deliberately no resume tool."""

from __future__ import annotations

import json
import threading
import time
import uuid

from strands import Agent, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager

from iotlink import Link

SYSTEM = (
    "You coordinate a robot fleet. list_devices shows each robot's zone and skills; describe gives its joints, "
    'limits, and what each skill does. Map each part of the goal to a robot in the right zone that offers the '
    'skill, start it with run_skill (or act for an explicit joint move), and poll skill_status until it ends. '
    'Robots may work in parallel. If a robot stops answering mid-task its outcome is UNKNOWN: say so, then '
    'reassign to a robot in the SAME zone that offers the skill, or report it undone. A skill acts where its '
    'robot stands. A finished skill may report a check (its postcondition): if check.ok is false the task is NOT '
    'done; say what the check saw. You can e-stop; you cannot resume: only a human operator can. For several '
    'skills on one robot in order, use run_sequence (one call per robot; robots may run their sequences in '
    'parallel). After starting single skills, call wait_for_skill on each robot (not skill_status in a loop) and '
    'report only after every skill you started has ended. awaiting_approval means a human operator is deciding: '
    'it is not a failure; never re-send the skill or start another on that robot. Report what no robot can do. '
    'Report only what the tools said. Be short.'
)


class Fleet:
    def __init__(self, tenant: str = "site"):
        self.tenant, self.present, self.calls = tenant, {}, []  # the ledger: every tool call as it happens
        self.link = Link(f"planner-{uuid.uuid4().hex[:12]}", tenant)
        # Each robot publishes its description once, retained. Changes arrive here; an empty payload means it left.
        self.link.subscribe(f"{tenant}/presence/+", self._on_presence)
        # IoT Core's own lifecycle event when a robot's connection drops (robot client ids are robot-<device>)
        self.link.subscribe("$aws/events/presence/disconnected/+", self._on_disconnected)
        self._seed()  # after subscribing, so nothing published in between is missed
        self._beat = threading.Event()
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def _on_presence(self, topic: str, body: dict) -> None:
        if body:
            self.present[body["device"]] = body
        else:
            self.present.pop(topic.rsplit("/", 1)[-1], None)

    def _on_disconnected(self, _topic: str, body: dict) -> None:
        cid = str(body.get("clientId", ""))
        if cid.startswith("robot-"):
            self.present.pop(cid[len("robot-") :], None)

    def _seed(self) -> None:
        """Read the retained descriptions already there. IoT Core does not deliver a retained message to a wildcard
        subscription, so the planner asks for them through the IoT data API instead."""
        import boto3

        data = boto3.client("iot-data", region_name=self.link.region, endpoint_url=f"https://{self.link.endpoint}")
        prefix = f"{self.tenant}/presence/"
        for page in data.get_paginator("list_retained_messages").paginate():
            for m in page.get("retainedTopics", []):
                if m["topic"].startswith(prefix) and m.get("payloadSize", 0) > 0:
                    body = json.loads(data.get_retained_message(topic=m["topic"])["payload"])
                    self.present.setdefault(body["device"], body)

    def _heartbeat(self) -> None:  # the robots' watchdog holds any motion this planner started if this stops
        while not self._beat.is_set():
            self.link.publish(f"{self.tenant}/heartbeat/{self.link.client_id}", {"ts": time.time()})
            time.sleep(0.5)

    def devices(self, min_count: int = 1, timeout: float = 5.0) -> list:
        """Robots with a description on the fleet now; waits up to `timeout` for at least `min_count`."""
        end = time.time() + timeout
        while True:
            now = list(self.present.values())
            if len(now) >= min_count or time.time() > end:
                return now
            time.sleep(0.05)

    def tools(self) -> list:
        def call(device, rpc, **args):
            entry = {"t": round(time.time(), 2), "device": device, "rpc": rpc, **args}
            self.calls.append(entry)
            try:
                out = self.link.call(device, rpc, **args)
                entry["ok"] = True
                return out
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"[:200]
                raise

        @tool
        def list_devices() -> str:
            """Robots on the fleet now: device, zone, skills, estopped."""
            self.calls.append({"t": round(time.time(), 2), "device": None, "rpc": "list_devices"})
            return json.dumps([{k: d[k] for k in ("device", "zone", "skills", "estopped")} for d in self.devices()])

        @tool
        def describe(device: str) -> str:
            """A robot's manifest: joints with limits, each skill's duration and what it does, and its rpcs.

            Args:
                device: the robot id.
            """
            return json.dumps(call(device, "describe"))

        @tool
        def run_skill(device: str, skill: str) -> str:
            """Start one of the robot's own skills ON the robot. Returns at once; poll skill_status.

            Args:
                device: the robot id.
                skill: a skill the robot lists.
            """
            return json.dumps(call(device, "skill", name=skill))

        @tool
        def act(device: str, targets: dict, seconds: float) -> str:
            """Move named joints to targets over `seconds`, on the robot, in the units its describe() limits use.
            Out-of-range targets are refused, and a robot that runs approved skills only refuses this.

            Args:
                device: the robot id.
                targets: joint name -> target, in the robot's own joint units (see describe).
                seconds: how long the move takes, at most 30.
            """
            return json.dumps(call(device, "act", targets=targets, seconds=seconds))

        @tool
        def skill_status(device: str) -> str:
            """The robot's current or last motion: name, running, stopped_by.

            Args:
                device: the robot id.
            """
            return json.dumps(call(device, "status"))

        @tool
        def wait_for_skill(device: str, timeout_s: float = 180.0) -> str:
            """Wait until the robot's current skill has ended (or the timeout), polling the robot once a second.
            Use this instead of calling skill_status in a loop. A skill awaiting operator approval is still pending.

            Args:
                device: the robot id.
                timeout_s: give up waiting after this many seconds (the skill keeps its own bound on the robot).
            """
            end, last = time.time() + float(timeout_s), None
            while time.time() < end:
                last = self.link.call(device, "status")
                if not last.get("running") and last.get("state") != "awaiting_approval":
                    break
                time.sleep(1.0)
            self.calls.append({"t": round(time.time(), 2), "device": device, "rpc": "wait_for_skill", "final": last})
            return json.dumps(last)

        @tool
        def run_sequence(device: str, skills: list[str], timeout_s: float = 180.0) -> str:
            """Run several of one robot's skills in order: each starts after the previous one has ended. Code does the
            looping, not you: it stops at the first skill that fails, faults, or is refused, and returns one result per
            skill it attempted.

            Args:
                device: the robot id.
                skills: skill names from that robot's list, in the order to run them.
                timeout_s: the longest to wait for any one skill to end.
            """
            results = []
            for name in skills:
                try:
                    call(device, "skill", name=name)
                except Exception as exc:
                    results.append({"skill": name, "started": False, "error": str(exc)[:160]})
                    break
                end, st = time.time() + float(timeout_s), None
                while time.time() < end:
                    st = self.link.call(device, "status")
                    if not st.get("running") and st.get("state") != "awaiting_approval":
                        break
                    time.sleep(0.5)
                results.append(
                    {
                        "skill": name,
                        "started": True,
                        "stopped_by": (st or {}).get("stopped_by"),
                        "check": (st or {}).get("check"),
                    }
                )
                if (st or {}).get("stopped_by") != "done" or ((st or {}).get("check") or {}).get("ok") is False:
                    break
            return json.dumps(
                {
                    "device": device,
                    "requested": len(skills),
                    "ran": sum(r["started"] for r in results),
                    "results": results,
                }
            )

        @tool
        def emergency_stop(device: str) -> str:
            """Latch an e-stop on one robot. Only a human operator can resume it.

            Args:
                device: the robot id.
            """
            return json.dumps(call(device, "estop"))

        return [list_devices, describe, run_skill, run_sequence, act, skill_status, wait_for_skill, emergency_stop]

    def close(self) -> None:
        self._beat.set()
        self.link.close()


def plan(goal: str, model: str = "us.anthropic.claude-sonnet-5-5", min_devices: int = 1) -> dict:
    fleet, t0 = Fleet(), time.perf_counter()
    try:
        found = fleet.devices(min_count=int(min_devices))
        t_disc = time.perf_counter() - t0
        # Pin the goal: the default window keeps the last 40 messages, and a long order (approvals, waits) can push the
        # user's goal itself out of the model's context.
        # pin_first needs strands-agents 1.43 or later; an older install fails here with a TypeError.
        agent = Agent(
            model=model,
            system_prompt=SYSTEM,
            tools=fleet.tools(),
            callback_handler=None,
            conversation_manager=SlidingWindowConversationManager(window_size=200, pin_first=1),
        )
        reply = str(agent(goal)).strip()
        calls = [{"tool": c["rpc"], **{k: v for k, v in c.items() if k != "rpc"}} for c in fleet.calls]
        errors = [
            b["toolResult"]["content"][0].get("text", "")[:200]
            for m in agent.messages
            for b in m["content"]
            if "toolResult" in b and b["toolResult"].get("status") == "error"
        ]
        sent = {}
        for c in fleet.calls:  # what each robot was actually sent, from the ledger, not from the model's account
            if c["rpc"] == "skill" and c.get("ok"):
                sent.setdefault(c["device"], []).append(c.get("name"))
        return {
            "reply": reply,
            "devices": [d["device"] for d in found],
            "calls": calls,
            "errors": errors,
            "ledger_skills_sent": sent,
            "discover_s": round(t_disc, 2),
            "total_s": round(time.perf_counter() - t0, 2),
        }
    finally:
        fleet.close()
