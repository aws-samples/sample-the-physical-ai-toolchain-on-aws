"""AWS clients and phase execution shared by an Arena deployment's resource steps."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..source import REPOSITORY as REPO_ROOT


def module(relative):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location("arena_" + path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


support = module("isaac-lab-arena-on-aws/scripts/resources/support.py")
save, load = support.save, support.load


class Environment:
    """Run resource operations against the account in the saved deployment record."""

    @property
    def config(self):
        if self.record is None:
            raise ValueError(f"Deployment {self.name!r} is not prepared; use pai arena deploy")
        return self.record["config"]

    @property
    def profile_args(self):
        profile = self.config.get("profile")
        return ["--profile", profile] if profile else []

    @property
    def python(self):
        return sys.executable

    def child_environment(self):
        env = dict(os.environ, AWS_DEFAULT_REGION=self.config["region"],
                   AWS_REGION=self.config["region"])
        if self.config.get("profile"):
            for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                        "AWS_SECURITY_TOKEN", "AWS_DEFAULT_PROFILE"):
                env.pop(key, None)
            env["AWS_PROFILE"] = self.config["profile"]
        return env

    def client(self, service):
        import boto3
        from botocore.config import Config

        if self._session is None:
            session = boto3.Session(profile_name=self.config.get("profile"),
                                    region_name=self.config["region"])
            options = Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 3})
            actual = session.client("sts", config=options).get_caller_identity()["Account"]
            if actual != self.config["account_id"]:
                raise ValueError(f"Credentials select account {actual}; expected {self.config['account_id']}")
            self._session, self._api_options = session, options
        return self._session.client(service, config=self._api_options)

    def run(self, arguments, label):
        return support.run(arguments, repo=REPO_ROOT, logs=self.root / "logs",
                           name=label, env=self.child_environment())

    def terraform(self, name, arguments, label):
        executable = shutil.which("terraform")
        if not executable:
            raise ValueError("Terraform is missing; install Terraform before deploying")
        directory = self.states.get(name, self.root / name)
        return self.run([executable, "-chdir=" + str(directory), *arguments], label)

    def terraform_json(self, name, *arguments):
        executable = shutil.which("terraform")
        if not executable:
            raise ValueError("Terraform is missing; install Terraform before deploying")
        directory = self.states.get(name, self.root / name)
        output = subprocess.check_output([executable, "-chdir=" + str(directory), *arguments],
                                         cwd=REPO_ROOT, env=self.child_environment(), text=True)
        return json.loads(output)

    def outputs(self, name):
        return {k: v["value"] for k, v in self.terraform_json(name, "output", "-json").items()}

    def phase(self, name, action):
        previous = self.record.setdefault("phases", {}).get(name, {})
        if previous.get("status") == "Complete":
            print(f"{name}: already complete ({previous['seconds']:.1f}s); reusing saved work", flush=True)
            return
        started = time.monotonic()
        entry = {"status": "Running", "started_at": datetime.now(timezone.utc).isoformat(),
                 "prior_attempt": previous or None}
        self.record["phases"][name] = entry
        self.persist()
        print(f"\n{name}: starting", flush=True)
        try:
            action(self)
        except BaseException as exc:
            entry.update(status="Failed", error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            entry["status"] = "Complete"
        finally:
            entry.update(seconds=round(time.monotonic() - started, 1),
                         finished_at=datetime.now(timezone.utc).isoformat())
            self.persist()
            print(f"{name}: {entry['status']} after {entry['seconds']:.1f}s", flush=True)
