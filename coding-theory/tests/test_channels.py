import numpy as np
import pytest

from codingtheory.channels import (
    BSCChannel,
    BurstChannel,
    SymbolErrorChannel,
    bits_to_symbols,
    symbols_to_bits,
)


def test_bits_symbols_roundtrip():
    rng = np.random.default_rng(0)
    symbols = rng.integers(0, 16, size=(10, 15))
    assert (bits_to_symbols(symbols_to_bits(symbols, 4), 4) == symbols).all()


def test_bsc_zero_p_identity():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=(5, 100)).astype(np.uint8)
    assert (BSCChannel(0.0, rng).transmit(bits) == bits).all()


def test_bsc_flip_rate():
    rng = np.random.default_rng(2)
    zeros = np.zeros((100, 1000), dtype=np.uint8)
    rate = BSCChannel(0.1, rng).transmit(zeros).mean()
    assert abs(rate - 0.1) < 0.005


def test_symbol_channel_changes_whole_symbols():
    rng = np.random.default_rng(3)
    zeros = np.zeros((100, 60), dtype=np.uint8)
    out = SymbolErrorChannel(0.2, 4, rng).transmit(zeros)
    sym = bits_to_symbols(out, 4)
    rate = (sym > 0).mean()
    assert abs(rate - 0.2) < 0.02


def test_burst_contiguous_and_bounded():
    rng = np.random.default_rng(4)
    zeros = np.zeros((500, 60), dtype=np.uint8)
    out = BurstChannel(1.0, 8, rng).transmit(zeros)
    for frame in out:
        idx = np.flatnonzero(frame)
        assert 1 <= len(idx) <= 8
        assert idx[-1] - idx[0] + 1 == len(idx)  # непрерывность


def test_burst_probability():
    rng = np.random.default_rng(5)
    zeros = np.zeros((5000, 60), dtype=np.uint8)
    out = BurstChannel(0.25, 8, rng).transmit(zeros)
    rate = (out.sum(axis=1) > 0).mean()
    assert abs(rate - 0.25) < 0.03


def test_invalid_params():
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError):
        BSCChannel(1.5, rng)
    with pytest.raises(ValueError):
        BurstChannel(0.1, 0, rng)
