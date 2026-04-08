"""Experimental integrity logging (CLAUDE.md §6.1 — Cherry Picking 금지).

ALL training runs are logged, including:
- Successful runs
- Divergent runs (solver failures)
- Early-stopped runs
- Error runs

No post-hoc filtering. Divergence rate is tracked as a first-class metric.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunLogger:
    """Logger for a single training run. Writes JSON lines.

    Every run gets a UUID. All hyperparameters, epoch metrics, and final
    outcomes are recorded regardless of success/failure.

    Usage:
        logger = RunLogger("logs/runs.jsonl", seed=42, hparams={...})
        for epoch in range(n_epochs):
            logger.log_epoch(epoch, {"loss": 0.5, "divergence_rate": 0.02})
        logger.finalize("success", {"test_aafe": 2.1})
    """

    def __init__(
        self,
        log_path: str | Path,
        seed: int,
        hparams: Dict[str, Any],
        run_id: Optional[str] = None,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.seed = seed
        self.hparams = hparams
        self.start_time = datetime.utcnow().isoformat()
        self.epoch_history: List[Dict[str, Any]] = []
        self.status = "running"

        # Log run start
        self._write({
            "event": "run_start",
            "run_id": self.run_id,
            "seed": self.seed,
            "hparams": self.hparams,
            "timestamp": self.start_time,
        })

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a single epoch."""
        entry = {"epoch": epoch, **metrics}
        self.epoch_history.append(entry)
        self._write({
            "event": "epoch",
            "run_id": self.run_id,
            "epoch": epoch,
            "metrics": metrics,
        })

    def log_divergence(
        self,
        epoch: int,
        mol_id: str,
        error_msg: str,
    ) -> None:
        """Log a solver divergence event (CLAUDE.md §6.1 — Solver Divergence Transparency).

        Divergent molecules are NEVER silently dropped.
        """
        self._write({
            "event": "divergence",
            "run_id": self.run_id,
            "epoch": epoch,
            "mol_id": mol_id,
            "error": error_msg,
        })

    def finalize(
        self,
        status: str,
        final_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finalize the run. Status: "success", "diverged", "early_stopped", "error".

        Args:
            status: Terminal status of the run.
            final_metrics: Summary metrics (test loss, AAFE, divergence_rate, etc.).
        """
        self.status = status
        self._write({
            "event": "run_end",
            "run_id": self.run_id,
            "status": status,
            "final_metrics": final_metrics or {},
            "n_epochs": len(self.epoch_history),
            "start_time": self.start_time,
            "end_time": datetime.utcnow().isoformat(),
        })

    def _write(self, record: Dict[str, Any]) -> None:
        """Append a JSON line to the log file."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
