"""Workflow state for MLE-STAR Deep Agents edition."""

import time
from typing import Optional
from pydantic import BaseModel, Field


class WorkflowState(BaseModel):
    """State that flows between phases via the main agent's context."""
    phase: str = "initialization"
    start_time: float = Field(default_factory=time.time)
    task_description: str = ""

    # Phase 1: Initialization
    retrieved_models: list[dict] = Field(default_factory=list)
    evaluated_models: list[dict] = Field(default_factory=list)
    initial_solution: str = ""
    initial_score: Optional[float] = None

    # Phase 2: Refinement (L parallel solutions)
    refined_solutions: list[str] = Field(default_factory=list)
    refined_scores: list[float] = Field(default_factory=list)

    # Phase 3: Ensemble
    ensemble_solution: str = ""
    ensemble_score: Optional[float] = None

    # Final
    final_solution: str = ""
    final_score: Optional[float] = None
    submission_file: Optional[str] = None

    # Budget
    sandbox_call_count: int = 0
    sandbox_call_limit: int = 100

    @property
    def sandbox_calls_remaining(self) -> int:
        return max(0, self.sandbox_call_limit - self.sandbox_call_count)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def best_solution(self) -> str:
        """Return the best solution available so far."""
        if self.final_solution:
            return self.final_solution
        if self.ensemble_solution:
            return self.ensemble_solution
        if self.refined_solutions:
            best_idx = self.refined_scores.index(max(self.refined_scores)) if self.refined_scores else 0
            return self.refined_solutions[best_idx]
        return self.initial_solution

    @property
    def best_score(self) -> Optional[float]:
        """Return the best score achieved so far."""
        scores = []
        if self.final_score is not None:
            scores.append(self.final_score)
        if self.ensemble_score is not None:
            scores.append(self.ensemble_score)
        if self.refined_scores:
            scores.append(max(self.refined_scores))
        if self.initial_score is not None:
            scores.append(self.initial_score)
        return max(scores) if scores else None
