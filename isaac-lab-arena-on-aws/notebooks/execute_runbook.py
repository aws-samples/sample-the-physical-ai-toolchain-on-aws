"""Execute the same runbook cells as Jupyter Run All, saving progress in place.

Usage: python notebooks/execute_runbook.py
The notebook reads config.json itself; this runner never rewrites inputs.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import uuid

import nbformat
from nbclient import NotebookClient


def now():
    return datetime.now(timezone.utc).isoformat()


class SavingClient(NotebookClient):
    def __init__(self, notebook, path, events, **kwargs):
        super().__init__(notebook, **kwargs)
        self.path, self.events = path, events
        self.last_save = 0.0
        self.on_cell_start = self.cell_started
        self.on_cell_executed = self.cell_finished

    def event(self, kind, **details):
        entry = {"at": now(), "event": kind, **details}
        with self.events.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    def save_progress(self):
        self.nb.metadata["runbook_execution"]["last_saved_at"] = now()
        temporary = self.path.with_name("." + self.path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            nbformat.write(self.nb, temporary)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.last_save = time.monotonic()

    def cell_started(self, cell, cell_index):
        if cell.cell_type != "code":
            return
        cell.metadata["runbook_status"] = "running"
        self.nb.metadata["runbook_execution"].update(active_cell=cell_index, status="running")
        self.event("cell-started", index=cell_index, id=cell.id)
        self.save_progress()

    def cell_finished(self, cell, cell_index, execute_reply):
        status = "completed" if execute_reply["content"]["status"] == "ok" else "failed"
        cell.metadata["runbook_status"] = status
        self.event("cell-finished", index=cell_index, id=cell.id, status=status)
        self.save_progress()

    def process_message(self, msg, cell, cell_index):
        try:
            return super().process_message(msg, cell, cell_index)
        finally:
            if time.monotonic() - self.last_save >= 5:
                self.save_progress()


async def execute(path, kernel_name=None):
    path = path.resolve()
    notebook = nbformat.read(path, as_version=4)
    output = path.parent.parent / "local-dev/notebook-executions" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    # Preserve prior outputs before a human-authorized rerun, without switching
    # to a different instruction notebook or changing any cell source.
    shutil.copyfile(path, output / "before.ipynb")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
            cell.metadata.pop("runbook_status", None)
    notebook.metadata["runbook_execution"] = {
        "status": "starting", "started_at": now(), "active_cell": None,
        "events": str(output / "events.jsonl"),
    }
    client = SavingClient(
        notebook, path, output / "events.jsonl", timeout=None,
        kernel_name=kernel_name or notebook.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.event("started", notebook=str(path))
    client.save_progress()

    async def heartbeat():
        while True:
            await asyncio.sleep(5)
            client.save_progress()

    saving = asyncio.create_task(heartbeat())
    try:
        await client.async_execute()
    except BaseException as exc:
        notebook.metadata["runbook_execution"].update(
            status="interrupted" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "failed",
            error=f"{type(exc).__name__}: {exc}", finished_at=now(),
        )
        client.event("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        notebook.metadata["runbook_execution"].update(status="completed", active_cell=None, finished_at=now())
        client.event("completed", notebook=str(path))
    finally:
        saving.cancel()
        try:
            await saving
        except asyncio.CancelledError:
            pass
        client.save_progress()
        shutil.copyfile(path, output / "finished.ipynb")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path, default=Path(__file__).with_name("runbook.ipynb"))
    parser.add_argument("--kernel", help="Use an installed Jupyter kernel instead of the notebook's default.")
    args = parser.parse_args()
    asyncio.run(execute(args.notebook, args.kernel))


if __name__ == "__main__":
    main()
