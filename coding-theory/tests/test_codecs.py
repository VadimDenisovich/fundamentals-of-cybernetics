import numpy as np
import pytest

from codingtheory.codecs import GaloisBCH, GaloisRS, ReedsoloRS

CODECS = [
    lambda: GaloisRS(15, 11),
    lambda: GaloisBCH(15, 7),
    lambda: GaloisRS(255, 223),
    lambda: ReedsoloRS(255, 223),
]


@pytest.mark.parametrize("make_codec", CODECS)
def test_roundtrip_no_errors(make_codec):
    codec = make_codec()
    rng = np.random.default_rng(10)
    msgs = codec.random_messages(20, rng)
    assert (codec.decode(codec.encode(msgs)) == msgs).all()


@pytest.mark.parametrize("make_codec", CODECS)
def test_corrects_up_to_t_symbol_errors(make_codec):
    codec = make_codec()
    rng = np.random.default_rng(11)
    msgs = codec.random_messages(20, rng)
    codewords = codec.encode(msgs)
    for row in codewords:
        positions = rng.choice(codec.n, size=codec.t, replace=False)
        for pos in positions:
            row[pos] ^= int(rng.integers(1, 2**codec.symbol_bits))
    assert (codec.decode(codewords) == msgs).all()


@pytest.mark.parametrize("make_codec", CODECS)
def test_systematic(make_codec):
    codec = make_codec()
    rng = np.random.default_rng(12)
    msgs = codec.random_messages(5, rng)
    codewords = codec.encode(msgs)
    assert (codewords[:, : codec.k] == msgs).all()
