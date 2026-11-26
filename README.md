
# StreamParity

[![PyPI version](https://img.shields.io/pypi/v/streamparity)](https://pypi.org/project/streamparity/)
[![CI](https://github.com/jaskaran04/streamparity/actions/workflows/ci.yml/badge.svg)](https://github.com/jaskaran04/streamparity/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/streamparity/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jaskaran04/streamparity/blob/main/LICENSE)
[![Development Status](https://img.shields.io/badge/status-alpha-orange)](https://pypi.org/project/streamparity/)
[![Downloads](https://img.shields.io/pypi/dm/streamparity)](https://pypi.org/project/streamparity/)

**Research-to-production consistency testing for numerical and streaming pipelines.**

StreamParity is an open-source Python testing framework for validating consistency between batch calculations and incremental, stateful implementations.

It helps researchers, quantitative developers and software engineers identify discrepancies between historical simulations and production-style replay, including lookahead bias, chunk-dependent behaviour, checkpoint recovery failures and inconsistent downstream decisions.

StreamParity is framework-agnostic, requires only NumPy at runtime and can be integrated into existing automated testing and CI/CD pipelines.

## Installation

StreamParity is available on the official Python Package Index (PyPI).

```sh
pip install streamparity
```

Requires Python 3.10+ and NumPy.

- [PyPI](https://pypi.org/project/streamparity/)
- [GitHub](https://github.com/jaskaran04/streamparity)
- [Issue Tracker](https://github.com/jaskaran04/streamparity/issues)

## Why StreamParity?

Numerical algorithms are frequently developed using historical datasets processed in batches, while production systems receive observations incrementally.

Even when two implementations are intended to represent the same algorithm, differences in state management, data partitioning, information availability and numerical precision can produce inconsistent results.

StreamParity provides a unified testing interface for detecting these discrepancies before deployment.

### Applications

| Domain | Application |
| --- | --- |
| Quantitative trading | Backtest-to-live consistency and lookahead bias detection |
| Derivatives and risk analytics | Numerical pipeline and stateful risk calculation consistency |
| Portfolio management | Incremental portfolio and P&L calculation validation |
| Machine learning | Offline-to-online feature transformation consistency |
| Fraud detection | Batch-to-stream scoring consistency |
| Scientific computing | Numerical algorithm regression testing |
| Sensor processing | Stateful calculation and checkpoint recovery validation |
| Data engineering | Chunk-invariant numerical transformations |

## Features

- **Batch-stream equivalence:** Compare historical batch calculations with incremental replay.
- **Lookahead bias detection:** Identify violations of prefix consistency that may indicate dependence on future observations.
- **Fixed chunk invariance:** Detect sensitivity to input partition sizes.
- **Randomized chunk invariance:** Test reproducible random partition schedules.
- **Checkpoint recovery:** Identify discrepancies caused by incomplete or incorrect state restoration.
- **Decision-level validation:** Detect exact decision mismatches that numerical tolerance tests may overlook.
- **Structured reporting:** Export audit results as dictionaries or JSON for automated analysis.
- **CI/CD integration:** Enforce consistency checks through automated test assertions.
- **Minimal dependencies:** NumPy is the only runtime dependency.

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


report = audit(
    np.cumsum,
    streaming,
    [1., -2., 3., -1.],
    random_trials=5,
    seed=42,
    max_chunk=3,
    decision=lambda y: y > 0,
)

print(report.summary())

# Raises AssertionError if any check fails.
report.assert_passed()
```

## Available checks

| Function | Comparison |
| --- | --- |
| `check_batch_stream` | Batch output vs single-row incremental replay |
| `check_chunks` | Fixed-size partitions vs single-row replay |
| `check_lookahead` | Full-run historical outputs vs prefix-only executions |
| `check_random_chunks` | Seeded random partitions vs single-row replay |
| `check_restart` | Uninterrupted replay vs checkpoint/restart replay |
| `check_decisions` | Exact discrete decisions from batch vs single-row streaming outputs |
| `audit` | Combined checks; optional random, restart and decision checks |

## Audit reports

`AuditReport` provides the following interfaces:

```python
report.passed
report.checks
report.summary()
report.assert_passed()
report.to_dict()
report.to_json()
```

Each check reports:

- Test scenario.
- Pass or fail status.
- Number of mismatched output rows.
- Zero-based index of the first mismatch.

Shape mismatches are reported separately, with a row mismatch count of zero and a null first-mismatch index.

No raw inputs or output values are included in reports. User-selected scenarios are not exhaustive.

### Integration with automated testing

StreamParity audits can be incorporated directly into existing pytest test suites.

```python
def test_pipeline_consistency():
    report = audit(
        batch_model,
        streaming_factory,
        test_data,
        random_trials=20,
        seed=42,
    )

    report.assert_passed()
```

Here, `batch_model`, `streaming_factory` and `test_data` are supplied by the application being tested.

This allows developers to detect consistency regressions during continuous integration before deploying changes.

## Decision-level consistency

Numerical equivalence does not necessarily imply decision equivalence.

For example, a trading strategy may generate a positive or negative signal depending on whether its output exceeds zero.

| Implementation | Numerical output | Decision |
| --- | ---: | --- |
| Batch | +1e-12 | Buy |
| Streaming | -1e-12 | Sell |

These outputs may be considered numerically equivalent under a conventional floating-point tolerance, despite producing opposite trading decisions.

StreamParity supports exact comparison of discrete downstream decisions through a user-defined mapping.

```python
from streamparity import check_decisions

result = check_decisions(
    batch_model,
    streaming_factory,
    data,
    decision=lambda y: y > 0,
)
```

This is particularly relevant to threshold-based trading strategies, risk limits, classification systems and automated decision pipelines.

## Checkpoint recovery

A checkpoint callback serializes the updater's state to immutable bytes. A restore callback loads those bytes into a fresh updater in place.

Test the same serialization callbacks used by the real application whenever possible.

```python
from streamparity import check_restart

results = check_restart(
    streaming_factory,
    data,
    checkpoint=lambda updater: updater.save_state(),
    restore=lambda updater, state: updater.load_state(state),
    points=(10, 50),
)

assert all(r.passed for r in results)
```

The restart points specify the number of consumed rows after which the state is saved and restored.

See [`examples/trading_replay.py`](https://github.com/jaskaran04/streamparity/blob/main/examples/trading_replay.py) for a complete runnable implementation and an intentional recovery failure.

This simulates state restoration within a process. It does not terminate processes, simulate network failures or verify durability on disk.

## Trading demonstration

The repository includes a synthetic trading pipeline demonstrating consistency across batch and streaming execution, randomized partitions, discrete trading decisions and checkpoint recovery.

```sh
python examples/trading_replay.py
```

Results from the v0.2.0 release verification:

| Scenario | Checks passed |
| --- | ---: |
| Correct implementation | 18/18 |
| Implementation with broken recovery | 15/18 |

All three intentionally introduced checkpoint recovery failures are detected.

```text
FAIL restart [after=1]: 7 mismatched rows; first at 1
FAIL restart [after=4]: 4 mismatched rows; first at 4
FAIL restart [after=7]: 1 mismatched rows; first at 7
```

These results demonstrate defect detection on a controlled example, not performance on an institutional trading system.

## Contracts and numerical semantics

- `batch(data)` is deterministic and stateless or freshly reset. `streaming()` returns a fresh stateful callable `update(chunk)` for each replay. State is retained across calls to that updater only.

- Axis 0 is time. Both transforms return one nonempty numerical output row per input row, in identical order. Input and output arrays are copied to isolate ordinary in-place mutations.

- Real numeric and Boolean arrays are supported. Inputs must be nonempty. Lookahead, restart and combined audits require at least two rows.

- Integer and Boolean outputs compare exactly when their discrete representation is preserved throughout replay. Floating-point or mixed outputs use NumPy `isclose(actual, expected, rtol, atol, equal_nan)`.

- Mixed integer/floating-point comparisons may lose precision. Return matching discrete dtypes for exact decisions.

- Matching NaNs are equal by default. Set `equal_nan=False` to reject them. Equal infinities compare equal.

- `decision(outputs)` must be a deterministic row-wise mapping to integer or Boolean labels. It is evaluated separately on complete batch and incremental output arrays and compared exactly, rather than using numerical tolerances.

- StreamParity cannot verify the causality of a user-supplied decision function.

- Exceptions from user code propagate.

- Random trials use a nonnegative integer seed and bounded random partitions. A seed and scenario identify a replay within the same NumPy environment.

## Development and installation from source

Clone the repository:

```sh
git clone https://github.com/jaskaran04/streamparity.git
cd streamparity
```

Install the package and development dependencies:

```sh
python -m pip install -e ".[dev]"
```

Run the automated tests:

```sh
python -m pytest -v
```

Run the examples:

```sh
python examples/moving_sum.py
python examples/trading_replay.py
```

Build and validate the package:

```sh
python -m build
python -m twine check --strict dist/*
```

## Testing and release infrastructure

StreamParity v0.2.0 includes:

- 34 automated unit tests.
- Verification on Windows and Linux.
- GitHub Actions continuous integration.
- Automated wheel and source distribution builds.
- Strict package metadata validation.
- Separate TestPyPI and production PyPI publishing workflows.
- Secure distribution through PyPI Trusted Publishing.
- Clean-environment installation and functionality verification.

## Limitations

StreamParity is currently alpha software and an offline testing framework, not a certified production validation system.

Passing tests provide evidence of consistency on the supplied data, scenarios and numerical tolerances. They do not constitute proof of algorithmic equivalence, causality, absence of lookahead bias or profitable trading.

Future-derived features, externally fitted parameters and untested prefixes can evade lookahead checks.

Fixed and randomized partitions preserve input ordering.

### Known numerical limitations

Mixing signed and unsigned integer output dtypes across streaming chunks may cause NumPy to promote values to floating-point representations, potentially losing precision.

Use consistent output dtypes throughout replay, particularly for large integers and discrete decisions.

Inconsistent output dimensions between streaming chunks may raise a concatenation exception rather than produce a structured audit failure.

The current release does not support:

- Asynchronous or late-event replay.
- Timestamp-aware output alignment.
- DataFrame metadata preservation.
- Stochastic transformations.
- Distributed execution.
- Actual process-crash simulation.
- Verification of checkpoint durability.

The implementation replays and stores complete outputs. Benchmark memory consumption and execution time before large-scale use.

## Roadmap

Planned areas of development include:

- [ ] Timestamp-aware numerical comparisons.
- [ ] Missing, duplicate and out-of-order event simulation.
- [ ] Warm-up period sensitivity analysis.
- [ ] Floating-point precision and numerical stability diagnostics.
- [ ] Extended support for pandas and other numerical frameworks.
- [ ] Options pricing and risk analytics demonstrations.
- [ ] Large-scale performance and memory benchmarks.
- [ ] Expanded documentation and pytest integration.

These are proposed extensions and are not part of the current release.

## Contributing

Contributions, bug reports, feature requests and application examples are welcome.

Please open an issue or submit a pull request through the GitHub repository.

Changes to existing functionality should include corresponding regression tests.

For release preparation and publishing instructions, see `RELEASING.md`.

## Author and maintainer

**Jaskaran Singh**

- GitHub: https://github.com/jaskaran04
- PyPI project: https://pypi.org/project/streamparity/

## Licence

StreamParity is distributed under the MIT License.

See [LICENSE](https://github.com/jaskaran04/streamparity/blob/main/LICENSE) for details.
