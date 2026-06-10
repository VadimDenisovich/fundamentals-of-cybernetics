"""Направление А: сравнение реализаций RS(255,223,16) — galois vs reedsolo.

Идентичные условия: одна модель канала (BSC), одинаковые seed, p_range,
N_blocks и batch_size => обе библиотеки получают побайтово одинаковые
сообщения и реализации шума (кодовые слова идентичны, см. валидацию).

Запуск: python experiments/run_direction_a.py
"""

import pathlib

import numpy as np

from codingtheory import BSCChannel, GaloisRS, ReedsoloRS, run_sweep, save_csv

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"

N_BLOCKS = 10_000
SEED = 20260
BATCH = 2_000
P_RANGE = np.logspace(np.log10(0.004), np.log10(0.015), 8)


def main():
    RESULTS.mkdir(exist_ok=True)
    make_bsc = lambda p, rng: BSCChannel(p, rng)

    for codec, fname in [
        (GaloisRS(255, 223), "direction_a_galois.csv"),
        (ReedsoloRS(255, 223, fcr=1), "direction_a_reedsolo.csv"),
    ]:
        print(f"\n=== {codec.name}: N={N_BLOCKS}, seed={SEED} ===")
        rows = run_sweep(
            codec, make_bsc, P_RANGE, N_BLOCKS, SEED,
            batch_size=BATCH, channel_label="BSC",
        )
        save_csv(rows, RESULTS / fname)
        print(f"-> {RESULTS / fname}")


if __name__ == "__main__":
    main()
