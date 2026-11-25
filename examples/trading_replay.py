"""Synthetic cumulative P&L audit; no financial data or trading recommendations."""
import struct
import numpy as np
from streamparity import audit


class RunningPnL:
    def __init__(self):
        self.total = 0.0

    def __call__(self, chunk):
        out = self.total + np.cumsum(chunk)
        self.total = float(out[-1])
        return out

    def checkpoint(self):
        return struct.pack('!d', self.total)

    def restore(self, state):
        self.total, = struct.unpack('!d', state)


def main():
    pnl = np.array([0.5, -1.25, 2.0, -0.5, 0.25, -1.5, 1.0, 0.5])
    settings = dict(random_trials=5, seed=42, max_chunk=4,
                    decision=lambda y: y >= 0,
                    checkpoint=RunningPnL.checkpoint)
    good = audit(np.cumsum, RunningPnL, pnl, restore=RunningPnL.restore, **settings)
    print('Correct implementation:')
    print(good.summary())
    good.assert_passed()
    broken = audit(np.cumsum, RunningPnL, pnl, restore=lambda fresh, state: None, **settings)
    print('\nBroken recovery:')
    print(broken.summary())
    assert not broken.passed


if __name__ == '__main__':
    main()
