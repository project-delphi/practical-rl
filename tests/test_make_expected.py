"""The fixture drift check tolerates last-bit platform noise, NaN padding included."""

import numpy as np

import make_expected


def test_same_content_allows_platform_noise_and_nan_padding():
    old = make_expected.save_npz(v=np.array([1.0, np.nan, 3.0]), k=np.array([1, 2]))
    noisy = make_expected.save_npz(v=np.array([1.0 + 1e-15, np.nan, 3.0]), k=np.array([1, 2]))
    assert old != noisy  # not byte-identical, so the per-key comparison runs
    assert make_expected.same_content(old, noisy)
    moved = make_expected.save_npz(v=np.array([1.0, 2.0, 3.0]), k=np.array([1, 2]))
    assert not make_expected.same_content(old, moved)  # NaN is not a wildcard
    other_ints = make_expected.save_npz(v=np.array([1.0, np.nan, 3.0]), k=np.array([1, 3]))
    assert not make_expected.same_content(old, other_ints)
