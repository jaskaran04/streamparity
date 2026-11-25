import numpy as np
import pytest
from streamparity import audit, check_batch_stream, check_chunks, check_lookahead


def cumulative():
    total = 0.0
    def update(chunk):
        nonlocal total
        out = total + np.cumsum(chunk)
        total = out[-1]
        return out
    return update


X = np.arange(1., 11.)


def test_correct_pipeline():
    report = audit(np.cumsum, cumulative, X)
    assert report.passed
    report.assert_passed()


def test_missing_state():
    r = check_batch_stream(np.cumsum, lambda: np.cumsum, X)
    assert not r.passed and r.first_mismatch == 1


def test_chunk_reset():
    assert not check_chunks(lambda: np.cumsum, X, sizes=(2,))[0].passed


def test_lookahead():
    assert not all(r.passed for r in check_lookahead(lambda x: x-x.mean(), X))


def test_causal():
    assert all(r.passed for r in check_lookahead(np.cumsum, X))


def test_nan_policy():
    x = np.array([np.nan, 1.])
    assert check_batch_stream(lambda x:x, lambda: lambda x:x, x).passed
    assert not check_batch_stream(lambda x:x, lambda: lambda x:x, x, equal_nan=False).passed


def test_shape_mismatch():
    assert not check_batch_stream(lambda x:x[:,None], lambda: lambda x:x, X).passed


def test_discrete_exact():
    x = np.array([2**63, 2**63+1], dtype=np.uint64)
    assert not check_batch_stream(lambda x:x, lambda: lambda x:x+1, x).passed


def test_input_isolation():
    x = X.copy()
    def mutate(z):
        z[:] = 0
        return z
    assert check_batch_stream(mutate, lambda:mutate, x).passed
    np.testing.assert_array_equal(x, X)


@pytest.mark.parametrize('data', [[], 1, ['a']])
def test_invalid_input(data):
    with pytest.raises((ValueError, TypeError)):
        check_batch_stream(np.cumsum, cumulative, data)


def test_invalid_settings():
    with pytest.raises(ValueError): check_chunks(cumulative, X, sizes=(0,))
    with pytest.raises(ValueError): check_lookahead(np.cumsum, X, prefixes=(10,))
    with pytest.raises(ValueError): check_batch_stream(np.cumsum, cumulative, X, atol=-1)


def test_multivariate():
    x = np.arange(12).reshape(6,2)
    assert audit(lambda z:z*2, lambda:lambda z:z*2, x).passed


def test_assert_failure():
    report = audit(np.cumsum, lambda: np.cumsum, X)
    with pytest.raises(AssertionError): report.assert_passed()
