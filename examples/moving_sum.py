"""Run: python examples/moving_sum.py"""
import numpy as np
from streamparity import audit


def batch(x):
    return np.cumsum(x, dtype=float)


def streaming():
    total = 0.0

    def update(chunk):
        nonlocal total
        result = []
        for value in chunk:
            total += value
            result.append(total)
        return np.asarray(result)

    return update


if __name__ == "__main__":
    data = np.array([10, 20, 30, 40, 50], dtype=float)
    report = audit(batch, streaming, data)
    print(report.summary())
    report.assert_passed()
    # A deliberately non-causal research implementation.
    print(audit(lambda x: x - x.mean(), streaming, data).summary())
