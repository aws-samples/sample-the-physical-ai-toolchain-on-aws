"""Command spelling for help and recovery hints; execution is shared by aliases."""

from contextlib import contextmanager
from contextvars import ContextVar


_COMMAND = ContextVar("arena_command", default="vla")


def command_name():
    return _COMMAND.get()


@contextmanager
def command_context(prog):
    token = _COMMAND.set(prog)
    try:
        yield
    finally:
        _COMMAND.reset(token)
