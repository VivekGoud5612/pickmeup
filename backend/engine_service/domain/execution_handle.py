"""Execution-resource boundary for Engine runtime workers.

The engine service deliberately knows only this protocol.  Local threads,
processes, containers, and remote workers can each implement it without
changing the runtime domain model.
"""

from __future__ import annotations

from typing import Protocol


class ExecutionHandle(Protocol):
    """Controls the concrete resource used for an execution."""

    def start(self) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...
