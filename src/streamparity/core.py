"""One output row per input row; fresh state per replay."""
from dataclasses import asdict, dataclass
import json
from typing import Callable, Iterable
import numpy as np

Transform = Callable[[np.ndarray], np.ndarray]
Factory = Callable[[], Transform]


@dataclass(frozen=True)
class CheckResult:
    name: str
    scenario: str
    passed: bool
    mismatched_rows: int
    first_mismatch: int | None
    detail: str = ""


@dataclass(frozen=True)
class AuditReport:
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def summary(self) -> str:
        lines = [f"{'PASS' if c.passed else 'FAIL'} {c.name} [{c.scenario}]"
                 + (f": {c.detail}" if c.detail else "") for c in self.checks]
        return "\n".join(lines + [f"{sum(c.passed for c in self.checks)}/{len(self.checks)} passed"])

    def to_dict(self) -> dict:
        """Return JSON-compatible results (no raw customer data)."""
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}

    def to_json(self, *, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def assert_passed(self) -> None:
        if not self.passed:
            raise AssertionError(self.summary())


def _data(data):
    x = np.array(data, copy=True)
    if x.ndim == 0 or len(x) == 0:
        raise ValueError("data must have a nonempty time axis (axis 0)")
    if x.dtype.kind not in "biuf":
        raise TypeError("data must be real numeric or boolean")
    return x


def _output(value, n):
    y = np.array(value, copy=True)
    if y.ndim == 0 or len(y) != n or y.size == 0:
        raise ValueError("transform must return one nonempty output row per input row")
    if y.dtype.kind not in "biuf":
        raise TypeError("outputs must be real numeric or boolean")
    return y


def _tolerances(rtol, atol):
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("tolerances must be finite and nonnegative")


def _compare(name, scenario, expected, actual, rtol, atol, equal_nan):
    if expected.shape != actual.shape:
        return CheckResult(name, scenario, False, 0, None,
                           f"shape mismatch: {expected.shape} vs {actual.shape}")
    # Exact comparisons for discrete outputs avoid lossy integer-to-float conversion.
    if expected.dtype.kind in "biu" and actual.dtype.kind in "biu":
        if expected.dtype.kind == actual.dtype.kind or "u" not in (expected.dtype.kind, actual.dtype.kind):
            same = expected == actual
        else:
            same = expected.astype(object) == actual.astype(object)
    else:
        same = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)
    rows = np.all(same.reshape(len(expected), -1), axis=1)
    bad = np.flatnonzero(~rows)
    return CheckResult(name, scenario, not len(bad), len(bad),
                       int(bad[0]) if len(bad) else None,
                       f"{len(bad)} mismatched rows; first at {bad[0]}" if len(bad) else "")


def _replay(factory, x, size):
    return _run_schedule(factory(), x, (min(size, len(x)-i) for i in range(0, len(x), size)))


def _run_schedule(update, x, schedule):
    outputs = []
    start = 0
    for size in schedule:
        chunk = x[start:start+size].copy()
        outputs.append(_output(update(chunk), len(chunk)))
        start += size
    if start != len(x):
        raise ValueError("schedule must cover all rows exactly")
    return np.concatenate(outputs, axis=0)


def check_batch_stream(batch: Transform, streaming: Factory, data, *,
                       rtol=1e-7, atol=1e-10, equal_nan=True) -> CheckResult:
    """Compare a fresh batch execution with fresh one-row-at-a-time state."""
    x = _data(data)
    _tolerances(rtol, atol)
    expected = _output(batch(x.copy()), len(x))
    actual = _replay(streaming, x, 1)
    return _compare("batch_stream", "size=1", expected, actual, rtol, atol, equal_nan)


def check_chunks(streaming: Factory, data, *, sizes: Iterable[int] = (2, 8, 32),
                 rtol=1e-7, atol=1e-10, equal_nan=True) -> tuple[CheckResult, ...]:
    """Compare fresh streaming replays with the one-row baseline."""
    x = _data(data)
    _tolerances(rtol, atol)
    sizes = tuple(sizes)
    if not sizes or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 1 for s in sizes):
        raise ValueError("sizes must contain positive integers")
    baseline = _replay(streaming, x, 1)
    return tuple(_compare("chunks", f"size={s}", baseline, _replay(streaming, x, s),
                          rtol, atol, equal_nan) for s in sizes)


def check_lookahead(batch: Transform, data, *, prefixes: Iterable[int] | None = None,
                    rtol=1e-7, atol=1e-10, equal_nan=True) -> tuple[CheckResult, ...]:
    """Compare prefix-only outputs with matching rows from a full run.

    A pass is evidence on tested prefixes, not a proof of causality.
    """
    x = _data(data)
    _tolerances(rtol, atol)
    if len(x) < 2:
        raise ValueError("lookahead checks require at least two rows")
    if prefixes is None:
        prefixes = sorted({1, len(x)//4, len(x)//2, 3*len(x)//4, len(x)-1} - {0})
    prefixes = tuple(prefixes)
    if not prefixes or any(isinstance(p, bool) or not isinstance(p, (int, np.integer)) or not 1 <= p < len(x) for p in prefixes):
        raise ValueError("prefixes must be integers between 1 and len(data)-1")
    full = _output(batch(x.copy()), len(x))
    return tuple(_compare("lookahead", f"prefix={p}", full[:p],
                          _output(batch(x[:p].copy()), p), rtol, atol, equal_nan)
                 for p in prefixes)


def audit(batch: Transform, streaming: Factory, data, *, sizes=(2, 8, 32),
          prefixes=None, rtol=1e-7, atol=1e-10, equal_nan=True,
          random_trials=0, seed=0, max_chunk=32, decision=None,
          checkpoint=None, restore=None, restart_points=None) -> AuditReport:
    """Run all three checks; user transform exceptions propagate to the caller."""
    opts = dict(rtol=rtol, atol=atol, equal_nan=equal_nan)
    checks = [check_batch_stream(batch, streaming, data, **opts),
              *check_chunks(streaming, data, sizes=sizes, **opts),
              *check_lookahead(batch, data, prefixes=prefixes, **opts)]
    if isinstance(random_trials, bool) or not isinstance(random_trials, (int, np.integer)) or random_trials < 0:
        raise ValueError("random_trials must be a nonnegative integer")
    if random_trials:
        checks.extend(check_random_chunks(streaming, data, trials=random_trials,
                                          seed=seed, max_chunk=max_chunk, **opts))
    if decision is not None:
        checks.append(check_decisions(batch, streaming, data, decision=decision))
    if (checkpoint is None) != (restore is None):
        raise ValueError("checkpoint and restore must be provided together")
    if checkpoint is not None:
        checks.extend(check_restart(streaming, data, checkpoint=checkpoint,
                                    restore=restore, points=restart_points, **opts))
    elif restart_points is not None:
        raise ValueError("restart_points requires checkpoint and restore")
    return AuditReport(tuple(checks))


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def check_random_chunks(streaming: Factory, data, *, trials=10, seed=0, max_chunk=32,
                        rtol=1e-7, atol=1e-10, equal_nan=True) -> tuple[CheckResult, ...]:
    """Compare random chunk partitions against a fresh single-row replay.

    The seed makes scenarios reproducible within the same NumPy RNG environment.
    """
    x = _data(data)
    _tolerances(rtol, atol)
    trials = _positive_int(trials, "trials")
    max_chunk = _positive_int(max_chunk, "max_chunk")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    baseline = _replay(streaming, x, 1)
    results = []
    for trial in range(trials):
        remaining = len(x)
        schedule = []
        while remaining:
            size = int(rng.integers(1, min(max_chunk, remaining)+1))
            schedule.append(size)
            remaining -= size
        actual = _run_schedule(streaming(), x, schedule)
        # Full schedules can be enormous; provide seed/trial to reproduce instead.
        scenario = f"trial={trial},seed={seed},max_chunk={max_chunk}"
        results.append(_compare("random_chunks", scenario, baseline, actual,
                                rtol, atol, equal_nan))
    return tuple(results)


def check_restart(streaming: Factory, data, *, checkpoint, restore, points=None,
                  rtol=1e-7, atol=1e-10, equal_nan=True) -> tuple[CheckResult, ...]:
    """Compare uninterrupted single-row replay with checkpoint/restart at each point.

    checkpoint(update) returns immutable serialized state (bytes); restore(fresh, state)
    mutates a fresh updater in place. Each point is the number of rows already consumed.
    """
    x = _data(data)
    _tolerances(rtol, atol)
    if len(x) < 2:
        raise ValueError("restart checks require at least two rows")
    if points is None:
        points = sorted({1, len(x)//2, len(x)-1})
    points = tuple(points)
    if not points or any(isinstance(p, bool) or not isinstance(p, (int, np.integer)) or not 1 <= p < len(x) for p in points):
        raise ValueError("restart points must be integers between 1 and len(data)-1")
    baseline = _replay(streaming, x, 1)
    results = []
    for point in points:
        updater = streaming()
        head = _run_schedule(updater, x[:point], (1 for _ in range(point)))
        state = checkpoint(updater)
        if not isinstance(state, bytes):
            raise TypeError("checkpoint must return bytes")
        fresh = streaming()
        restore(fresh, state)
        tail = _run_schedule(fresh, x[point:], (1 for _ in range(len(x)-point)))
        actual = np.concatenate((head, tail), axis=0)
        results.append(_compare("restart", f"after={point}", baseline, actual,
                                rtol, atol, equal_nan))
    return tuple(results)


def check_decisions(batch: Transform, streaming: Factory, data, *, decision) -> CheckResult:
    """Compare exact discrete decisions, even when numerical outputs are close.

    decision(outputs) must be a deterministic row-wise mapping to bool/int labels.
    """
    x = _data(data)
    expected = _output(decision(_output(batch(x.copy()), len(x))), len(x))
    actual = _output(decision(_replay(streaming, x, 1)), len(x))
    if expected.dtype.kind not in "biu" or actual.dtype.kind not in "biu":
        raise TypeError("decision must return integer or boolean labels")
    return _compare("decisions", "size=1", expected, actual, 0, 0, False)

