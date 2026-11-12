"""The robot-side rules, tested without ROS or IoT. Each test is run twice: against the real Device (must pass) and
against a copy with that one rule removed (must fail), so each test is shown to be what catches the rule's absence.

  python -m pytest -q test_safety.py
"""

import time
import types

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

import robot_host
from robot_host import Device

# An exception that escapes a thread is a broken test, not a caught rule: fail on it instead of warning about it.
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")


class FakeArm:
    limits = {"j1": (-2.0, 2.0), "j2": (-1.5, 1.5)}

    def __init__(self):
        self.q, self.writes = {"j1": 0.0, "j2": 0.0}, 0

    def positions(self):
        return dict(self.q)

    def command(self, t):
        self.q.update(t)
        self.writes += 1

    def hold(self):
        pass


KEY = Ed25519PrivateKey.generate()
PUB = KEY.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def make(device_cls=Device):
    return device_cls(FakeArm(), "arm-x", "cell", {"wave": {"seconds": 0.2, "moves": [[{"j1": 0.5}, 0.2]]}}, PUB)


def wait_idle(d, s=3.0):
    end = time.monotonic() + s
    while d.task and d.task["running"] and time.monotonic() < end:
        time.sleep(0.01)


# --- the rules -------------------------------------------------------------------------------------------------
def rule_refuse_not_clamp(d):
    with pytest.raises(ValueError, match="outside its range"):
        d.act({"j1": 2.5}, 0.2, "planner")
    assert d.arm.writes == 0 and d.arm.q["j1"] == 0.0


def rule_estop_latches(d):
    d.estop("planner")
    with pytest.raises(PermissionError):
        d.act({"j1": 0.3}, 0.2, "planner")
    assert d.arm.writes == 0


def rule_resume_needs_operator_signature(d):
    d.estop("planner")
    for sig in ("", "00" * 64, KEY.sign(b"arm-x:some-other-stop").hex()):
        with pytest.raises(PermissionError):
            d.resume("planner", sig)
    assert d.estopped
    d.resume("safety-lead", KEY.sign(f"arm-x:{d.epoch}".encode()).hex())
    assert not d.estopped


def rule_watchdog_holds(d):
    d.act({"j1": 1.5}, 3.0, "planner")  # planner sends no heartbeat after starting
    wait_idle(d, 4.0)
    assert (d.task["stopped_by"] or "").startswith("watchdog") and d.arm.q["j1"] < 1.0


def rule_bounded_motion(d):
    with pytest.raises(ValueError, match="seconds"):
        d.act({"j1": 0.3}, 600, "planner")


RULES = [
    rule_refuse_not_clamp,
    rule_estop_latches,
    rule_resume_needs_operator_signature,
    rule_watchdog_holds,
    rule_bounded_motion,
]


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.__name__)
def test_rule_holds(rule):
    rule(make())


def test_a_check_that_cannot_measure_is_not_done_and_frees_the_robot():
    """Finished is not done: a check that raises is a failed check. It must not leave the skill "running" forever."""

    def broken():
        raise RuntimeError("camera offline")

    d = Device(FakeArm(), "arm-x", "cell", {"wave": {"moves": [[{"j1": 0.5}, 0.1]], "check": broken}}, PUB)
    d.skill("wave", "planner")
    wait_idle(d)
    assert d.task["running"] is False and d.task["stopped_by"] == "done"
    assert d.task["check"]["ok"] is False and "camera offline" in d.task["check"]["error"]
    d.act({"j1": 0.0}, 0.1, "planner")  # the robot takes its next command: it is not "busy with 'wave'"
    wait_idle(d)


def test_a_body_that_fails_mid_motion_ends_the_task_as_a_fault_and_frees_the_robot():
    """A body that raises while it moves (a bus error, or a body that moves by skills only, like the notebook's yard
    robots) must not leave the task "running" for good: the robot holds, says why, and takes its next command."""

    class SkillsOnly(FakeArm):
        holds = 0

        def command(self, t):
            raise NotImplementedError("this robot moves by skills only")

        def hold(self):
            self.holds += 1

    d = Device(SkillsOnly(), "duck-x", "yard-lane", {"wave": {"moves": [[{"j1": 0.5}, 0.1]]}}, PUB)
    d.act({"j1": 0.3}, 0.2, "planner")
    wait_idle(d)
    assert d.task["running"] is False
    assert d.task["stopped_by"] == "fault: NotImplementedError: this robot moves by skills only"
    assert d.arm.holds == 1 and "check" not in d.task
    with pytest.raises(LookupError):  # not "busy with 'act'": the robot answers the next command on its merits
        d.skill("fly", "planner")


# --- mutants: the same rule removed; its test must now fail ------------------------------------------------------
class NoRangeCheck(Device):
    def _start(self, name, moves, by, policy=None):  # clamps nothing, checks nothing: moves straight to the target
        self.arm.limits = {j: (-99, 99) for j in self.arm.limits}
        return super()._start(name, moves, by, policy)


class NoLatch(Device):
    def estop(self, by):
        self.arm.hold()
        return {"estopped": False}


class NameIsEnough(Device):
    def resume(self, operator, signature=""):
        self.estopped = False
        return {"estopped": False}


class NoWatchdog(Device):
    def _run(self, task, moves, policy=None):
        robot_host.QUIET_S, saved = 1e9, robot_host.QUIET_S
        try:
            super()._run(task, moves, policy)
        finally:
            robot_host.QUIET_S = saved


class Unbounded(Device):
    def _start(self, name, moves, by, policy=None):
        saved, robot_host.MAX_S = robot_host.MAX_S, 1e9
        try:
            return super()._start(name, moves, by, policy)
        finally:
            robot_host.MAX_S = saved


MUTANTS = {
    rule_refuse_not_clamp: NoRangeCheck,
    rule_estop_latches: NoLatch,
    rule_resume_needs_operator_signature: NameIsEnough,
    rule_watchdog_holds: NoWatchdog,
    rule_bounded_motion: Unbounded,
}


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.__name__)
def test_rule_test_catches_its_absence(rule):
    d = make(MUTANTS[rule])
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        rule(d)
    # The rule test must fail because the rule is missing, not because the mutant itself broke while moving.
    assert not str((d.task or {}).get("stopped_by") or "").startswith("fault"), d.task
    if d.task and d.task["running"]:
        d.estopped = True  # stop a mutant's runaway motion thread
