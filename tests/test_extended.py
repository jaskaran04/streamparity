import json
import struct
import numpy as np
import pytest
from streamparity import audit, check_random_chunks, check_restart, check_decisions

X = np.arange(1., 21.)

class Accumulator:
    def __init__(self): self.total = 0.
    def __call__(self, chunk):
        out = self.total + np.cumsum(chunk)
        self.total = out[-1]
        return out
    def checkpoint(self): return struct.pack('!d', self.total)
    def restore(self, state): self.total, = struct.unpack('!d', state)

def test_random_partitions_pass():
    assert all(c.passed for c in check_random_chunks(Accumulator, X, seed=17))

def test_random_partitions_fail_and_reproduce():
    a = check_random_chunks(lambda: np.cumsum, X, seed=4)
    b = check_random_chunks(lambda: np.cumsum, X, seed=4)
    assert a == b and any(not c.passed for c in a)

def test_random_single_row():
    assert all(c.passed for c in check_random_chunks(Accumulator, [1]))

@pytest.mark.parametrize('kwargs', [{'trials':0}, {'max_chunk':0}, {'seed':None}, {'seed':-1}])
def test_invalid_random_settings(kwargs):
    with pytest.raises(ValueError): check_random_chunks(Accumulator, X, **kwargs)

def test_restart_success():
    results = check_restart(Accumulator, X, checkpoint=Accumulator.checkpoint,
                            restore=Accumulator.restore, points=(1, 10, 19))
    assert all(c.passed for c in results)

def test_restart_lost_state():
    results = check_restart(Accumulator, X, checkpoint=Accumulator.checkpoint,
                            restore=lambda fresh, state: None, points=(7,))
    assert not results[0].passed and results[0].first_mismatch == 7

def test_restart_rejects_nonbytes():
    with pytest.raises(TypeError):
        check_restart(Accumulator, X, checkpoint=lambda u: u.total, restore=Accumulator.restore)

@pytest.mark.parametrize('points', [(), (0,), (20,), (True,)])
def test_bad_restart_points(points):
    with pytest.raises(ValueError):
        check_restart(Accumulator, X, checkpoint=Accumulator.checkpoint,
                      restore=Accumulator.restore, points=points)

def test_decisions_expose_small_error():
    r = audit(lambda x: x*0+1e-12, lambda: lambda x: x*0-1e-12, X,
              decision=lambda y: y>0)
    assert r.checks[0].passed and not r.checks[-1].passed
    assert r.checks[-1].mismatched_rows == len(X)

def test_decisions_exact_pass():
    assert check_decisions(np.cumsum, Accumulator, X, decision=lambda y:y>10).passed

def test_decisions_reject_float():
    with pytest.raises(TypeError): check_decisions(np.cumsum, Accumulator, X, decision=lambda y:y)

def test_combined_report_and_json():
    report = audit(np.cumsum, Accumulator, X, random_trials=4, decision=lambda x:x>10,
                   checkpoint=Accumulator.checkpoint, restore=Accumulator.restore)
    assert report.passed
    assert len(report.checks) == 17
    assert json.loads(report.to_json()) == report.to_dict()

def test_invalid_audit_settings():
    with pytest.raises(ValueError): audit(np.cumsum, Accumulator, X, checkpoint=Accumulator.checkpoint)
    with pytest.raises(ValueError): audit(np.cumsum, Accumulator, X, random_trials=-1)
    with pytest.raises(ValueError): audit(np.cumsum, Accumulator, X, restart_points=(1,))
