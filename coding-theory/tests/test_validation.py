import pytest

from codingtheory.validation import ALL_CHECKS


@pytest.mark.parametrize("name,check", ALL_CHECKS, ids=[name for name, _ in ALL_CHECKS])
def test_known_vector(name, check):
    ok, details = check()
    assert ok, f"{name}: {details}"
