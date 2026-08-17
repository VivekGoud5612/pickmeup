from __future__ import annotations

from dataclasses import dataclass, replace

"""
IMMUTATABLE RUN TIME STATE...
Value object representing the current progress of a training run.

TrainingProgress is immutable and captures a snapshot of
where a training session currently is.
"""

@dataclass(slots=True, frozen = True, kw_only=True)
class TrainingProgress:
    """
    Snapshot of training progress.
    """
    episode: int   ## Alreayd current as this is accessed by the current object....
    step: int
    total_episodes: int | None = None
    total_steps: int | None = None  # Let us move elapsed time to some monitoring, as here we just need training progress and nothing else...

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.episode < 0:
            raise ValueError("Current episode cannot be negative.")
        if self.step < 0:
            raise ValueError("Current step cannot be negative.")
        if (self.total_episodes is not None
            and self.total_episodes < 0):
            raise ValueError("Total episodes must be positive.")
        if (self.total_steps is not None
            and self.total_steps < 0):
            raise ValueError("Total steps must be positive.")

    @property   ## Production python code.... instead of progress.get_percentage, we can do progress.progress_percenage.it behaves like a field..
    def progress_percentage(self) -> float | None:
        """
        Percentage completion if total episodes are known.
        """
        if self.total_episodes is None:
            return None
        return (self.episode / self.total_episodes) * 100.0
        
    def advance_episode(self) -> "TrainingProgress":
        return replace(self, episode=self.episode + 1)
    
    def advance_step(self, amount : int) -> "TrainingProgress":
        return replace(self, step=self.step + amount)
    
