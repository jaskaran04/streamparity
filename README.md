# StreamParity

**Test research-to-production consistency in numerical streaming pipelines.**

StreamParity is an offline testing toolkit for comparing batch calculations with incremental, stateful implementations. Applications include quantitative trading, numerical risk models, sensor processing and production ML. It is currently an early-stage Python package, not a certified production validation system.

## Install from source

Requires Python 3.10+ and NumPy. From the repository root:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python examples/trading_replay.py
```

The name `streamparity` is proposed; public PyPI publication is not yet confirmed.

## Quick start

```python
import numpy as np
from streamparity import audit


def streaming():
    total = 0.0

    def update(chunk):
        nonlocal total
        out = total + np.cumsum(chunk)
        total = out[-1]
        return out

    return update


report = audit(np.cumsum, streaming, [1., -2., 3., -1.],
               random_trials=5, seed=42, max_chunk=3,
               decision=lambda y: y > 0)
print(report.summary())
report.assert_passed()  # raises AssertionError if a check fails
```

## Checks

| Function | Comparison |
| --- | --- |
| `check_batch_stream` | Batch output vs single-row incremental replay |
| `check_chunks` | Fixed-size partitions vs single-row replay |
| `check_lookahead` | Full-run historical outputs vs prefix-only executions |
| `check_random_chunks` | Seeded random partitions vs single-row replay |
| `check_restart` | Uninterrupted replay vs checkpoint/restart replay |
| `check_decisions` | Exact discrete decisions from batch vs single-row streaming outputs |
| `audit` | Combined checks; optional random, restart and decision checks |

`AuditReport` provides `.passed`, `.checks`, `.summary()`, `.assert_passed()`, `.to_dict()` and `.to_json()`. Each check reports its scenario, status, count of mismatched rows and zero-based first mismatch. Shape mismatches are reported separately (row mismatch count is zero and index is null). No raw inputs or output values are included in reports. User-selected scenarios are not exhaustive.

### Checkpoint recovery

A checkpoint callback serializes the updater's state to immutable bytes. A restore callback loads those bytes into a fresh updater in place. Test the same serialization callbacks used by the real application whenever possible.

```python
from streamparity import check_restart

results = check_restart(
    streaming_factory, data,
    checkpoint=lambda updater: updater.save_state(),
    restore=lambda updater, state: updater.load_state(state),
    points=(10, 50),  # restart after this many consumed rows
)
assert all(r.passed for r in results)
```

See `examples/trading_replay.py` for a complete runnable implementation and an intentional recovery failure. This simulates state restoration within a process; it does not kill processes, simulate network failures or verify durability on disk.

### Contracts and numerical semantics

- `batch(data)` is deterministic and stateless or freshly reset; `streaming()` returns a fresh stateful callable `update(chunk)` for each replay. State is retained across calls to that updater only.
- Axis 0 is time. Both transforms return one nonempty numerical output row per input row, in identical order. Input and output arrays are copied to isolate ordinary in-place mutations.
- Real numeric and boolean arrays are supported. Inputs must be nonempty; lookahead, restart and combined audits require at least two rows.
- Integer/boolean pairs compare exactly; floating or mixed outputs use NumPy `isclose(actual, expected, rtol, atol, equal_nan)`. Mixed integer/floating comparisons may lose precision; return matching discrete dtypes for exact decisions. Matching NaNs are equal by default; set `equal_nan=False` to reject them. Equal infinities compare equal.
- `decision(outputs)` must be a deterministic row-wise mapping to integer or boolean labels. It is evaluated separately on complete batch and incremental output arrays and compared exactly, not with numerical tolerances. The library cannot verify causality of the user-supplied decision function.
- Exceptions from user code propagate. Random trials use a nonnegative integer seed and bounded random partitions; a seed and scenario identify a replay within the same NumPy environment.

### Limits

Passing tests are evidence on the supplied data, scenarios and tolerances, not proof of equivalence, causality or profitable trading. Future-derived features, externally fitted parameters and untested prefixes can evade lookahead checks. Fixed and random partitions preserve input ordering. No asynchronous or late-event replay, timestamp alignment, dataframe metadata, stochastic transform support, distributed execution, or process-crash simulation. The current implementation replays and stores complete outputs; benchmark before large-scale use.

## Release preparation

See `RELEASING.md`. MIT licence is proposed; review the copyright attribution before distributing. Contributions should include regression tests.
