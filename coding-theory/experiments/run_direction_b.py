"""Направление Б: сравнение режимов на одной библиотеке (galois).

Вариант 1: разные коды с одинаковой избыточностью при n=15, t=2 —
RS(15,11,2) над GF(2^4) vs двоичный BCH(15,7,2) — в канале BSC.
Вариант 2: те же коды в пакетном канале (Burst, длина пакета <= 8 бит).
Дополнительно: RS(15,11) в канале символьных ошибок для сверки
с теоретической кривой FER (валидация Монте-Карло).

Запуск: python experiments/run_direction_b.py
"""

import pathlib

import numpy as np

from codingtheory import (
    BSCChannel,
    BurstChannel,
    GaloisBCH,
    GaloisRS,
    SymbolErrorChannel,
    run_sweep,
    save_csv,
)

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

N_BLOCKS = 100_000
SEED = 20261
BATCH = 25_000
BURST_MAX_LEN = 8
P_BSC = np.logspace(np.log10(0.003), np.log10(0.3), 10)
P_BURST = np.logspace(np.log10(0.001), np.log10(0.3), 10)
P_SYMBOL = np.logspace(np.log10(0.01), np.log10(0.3), 10)


def main():
    RESULTS.mkdir(exist_ok=True)
    rs = GaloisRS(15, 11)
    bch = GaloisBCH(15, 7)

    make_bsc = lambda p, rng: BSCChannel(p, rng)
    make_burst = lambda p, rng: BurstChannel(p, BURST_MAX_LEN, rng)

    sweeps = [
        (rs, make_bsc, P_BSC, "BSC", "direction_b_rs_bsc.csv"),
        (bch, make_bsc, P_BSC, "BSC", "direction_b_bch_bsc.csv"),
        (rs, make_burst, P_BURST, f"Burst(<= {BURST_MAX_LEN} бит)", "direction_b_rs_burst.csv"),
        (bch, make_burst, P_BURST, f"Burst(<= {BURST_MAX_LEN} бит)", "direction_b_bch_burst.csv"),
    ]
    for codec, make_channel, p_range, label, fname in sweeps:
        print(f"\n=== {codec.name} | {label}: N={N_BLOCKS}, seed={SEED} ===")
        rows = run_sweep(
            codec, make_channel, p_range, N_BLOCKS, SEED,
            batch_size=BATCH, channel_label=label,
        )
        save_csv(rows, RESULTS / fname)
        print(f"-> {RESULTS / fname}")

    # Валидация Монте-Карло: символьный канал, есть точная теоретическая FER
    make_symbol = lambda p, rng: SymbolErrorChannel(p, 4, rng)
    print(f"\n=== {rs.name} | SymbolError: N={N_BLOCKS}, seed={SEED} ===")
    rows = run_sweep(
        rs, make_symbol, P_SYMBOL, N_BLOCKS, SEED,
        batch_size=BATCH, channel_label="SymbolError",
    )
    save_csv(rows, RESULTS / "validation_rs_symbol.csv")
    print(f"-> {RESULTS / 'validation_rs_symbol.csv'}")


if __name__ == "__main__":
    main()
