"""Consistency checks for deterministic numerical pipelines."""
from .core import (
    AuditReport, CheckResult, audit, check_batch_stream, check_chunks,
    check_lookahead, check_random_chunks, check_restart, check_decisions,
)

__all__ = [
    "AuditReport", "CheckResult", "audit", "check_batch_stream", "check_chunks",
    "check_lookahead", "check_random_chunks", "check_restart", "check_decisions",
]
