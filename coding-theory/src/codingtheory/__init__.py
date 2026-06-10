"""Инструменты для сравнительного анализа помехоустойчивости кодов РС и BCH."""

from .channels import (
    BSCChannel,
    BurstChannel,
    SymbolErrorChannel,
    bits_to_symbols,
    symbols_to_bits,
)
from .codecs import GaloisBCH, GaloisRS, ReedsoloRS
from .metrics import two_proportion_ztest, wald_ci
from .simulate import run_sweep, save_csv, simulate_point

__all__ = [
    "BSCChannel",
    "BurstChannel",
    "SymbolErrorChannel",
    "bits_to_symbols",
    "symbols_to_bits",
    "GaloisBCH",
    "GaloisRS",
    "ReedsoloRS",
    "two_proportion_ztest",
    "wald_ci",
    "run_sweep",
    "save_csv",
    "simulate_point",
]
