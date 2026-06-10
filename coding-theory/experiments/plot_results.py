"""Построение графиков и статистических таблиц по CSV из results/.

Запуск: python experiments/plot_results.py
"""

import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from codingtheory.metrics import two_proportion_ztest
from codingtheory.simulate import load_csv

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
DPI = 160
FLOOR = 1e-7  # для отображения нулевых оценок на лог-оси


def _errbar(ax, rows, label, marker, metric="fer"):
    p = [r["p"] for r in rows]
    val = [max(r[metric], FLOOR) for r in rows]
    yerr_lo = [max(r[metric] - r[f"{metric}_ci_lo"], 0.0) for r in rows]
    yerr_hi = [max(r[f"{metric}_ci_hi"] - r[metric], 0.0) for r in rows]
    ax.errorbar(p, val, yerr=[yerr_lo, yerr_hi], label=label,
                marker=marker, capsize=3, linewidth=1.2, markersize=5)


def _setup(ax, xlabel, ylabel, title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def ztest_table(rows1, rows2, label1, label2, path, what="FER"):
    lines = [
        f"| p | {what} {label1} | {what} {label2} | z | p-value | различие |",
        "|---|---|---|---|---|---|",
    ]
    for r1, r2 in zip(rows1, rows2):
        assert abs(r1["p"] - r2["p"]) < 1e-12
        z, pv = two_proportion_ztest(
            int(r1["frame_errors"]), int(r1["n_blocks"]),
            int(r2["frame_errors"]), int(r2["n_blocks"]),
        )
        verdict = "значимо" if pv < 0.05 else "незначимо"
        lines.append(
            f"| {r1['p']:.4g} | {r1['fer']:.3e} | {r2['fer']:.3e} "
            f"| {z:+.2f} | {pv:.3f} | {verdict} |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"-> {path}")


def timing_table(rows_g, rows_r, path):
    lines = [
        "| p | galois, мс/блок | reedsolo, мс/блок | отношение | galois, кбит/с | reedsolo, кбит/с |",
        "|---|---|---|---|---|---|",
    ]
    ratios = []
    for rg, rr in zip(rows_g, rows_r):
        ratio = rr["decode_time_per_block_ms"] / rg["decode_time_per_block_ms"]
        ratios.append(ratio)
        lines.append(
            f"| {rg['p']:.4g} | {rg['decode_time_per_block_ms']:.3f} "
            f"| {rr['decode_time_per_block_ms']:.3f} | x{ratio:.1f} "
            f"| {rg['throughput_kbit_s']:.0f} | {rr['throughput_kbit_s']:.0f} |"
        )
    lines.append(f"\nСреднее отношение времени декодирования (reedsolo/galois): x{sum(ratios)/len(ratios):.1f}")
    path.write_text("\n".join(lines) + "\n")
    print(f"-> {path}")


def direction_a():
    rows_g = load_csv(RESULTS / "direction_a_galois.csv")
    rows_r = load_csv(RESULTS / "direction_a_reedsolo.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    _errbar(ax, rows_g, "galois RS(255,223)", "o")
    _errbar(ax, rows_r, "reedsolo RS(255,223)", "s")
    _setup(ax, "p (вероятность ошибки бита, BSC)", "FER",
           "Направление А: FER, galois vs reedsolo, RS(255,223), N=10⁴")
    fig.tight_layout()
    fig.savefig(RESULTS / "direction_a_fer.png", dpi=DPI)

    fig, ax = plt.subplots(figsize=(7, 5))
    _errbar(ax, rows_g, "galois RS(255,223)", "o", metric="ber")
    _errbar(ax, rows_r, "reedsolo RS(255,223)", "s", metric="ber")
    _setup(ax, "p (вероятность ошибки бита, BSC)", "BER (после декодирования)",
           "Направление А: BER, galois vs reedsolo, RS(255,223), N=10⁴")
    fig.tight_layout()
    fig.savefig(RESULTS / "direction_a_ber.png", dpi=DPI)

    fig, ax = plt.subplots(figsize=(7, 5))
    for rows, label, marker in [(rows_g, "galois", "o"), (rows_r, "reedsolo", "s")]:
        ax.plot([r["p"] for r in rows],
                [r["decode_time_per_block_ms"] for r in rows],
                marker=marker, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p (вероятность ошибки бита, BSC)")
    ax.set_ylabel("время декодирования, мс/блок")
    ax.set_title("Направление А: время декодирования на блок")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "direction_a_time.png", dpi=DPI)

    ztest_table(rows_g, rows_r, "galois", "reedsolo",
                RESULTS / "direction_a_ztest.md")
    timing_table(rows_g, rows_r, RESULTS / "direction_a_timing.md")


def direction_b():
    rs_bsc = load_csv(RESULTS / "direction_b_rs_bsc.csv")
    bch_bsc = load_csv(RESULTS / "direction_b_bch_bsc.csv")
    rs_burst = load_csv(RESULTS / "direction_b_rs_burst.csv")
    bch_burst = load_csv(RESULTS / "direction_b_bch_burst.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    _errbar(ax, rs_bsc, "RS(15,11,2), GF(2⁴)", "o")
    _errbar(ax, bch_bsc, "BCH(15,7,2)", "s")
    # теоретические пороги p ~ t/n
    ax.axvline(2 / 15, color="gray", linestyle="--", alpha=0.7,
               label="порог t/n = 2/15 (BCH, биты)")
    p_rs_threshold = 1 - (1 - 2 / 15) ** (1 / 4)  # битовый p, при котором p_s = t/n
    ax.axvline(p_rs_threshold, color="gray", linestyle=":", alpha=0.7,
               label=f"порог RS (p_s=t/n): p={p_rs_threshold:.3f}")
    _setup(ax, "p (вероятность ошибки бита, BSC)", "FER",
           "Направление Б: RS vs BCH в канале BSC, N=10⁵")
    fig.tight_layout()
    fig.savefig(RESULTS / "direction_b_bsc_fer.png", dpi=DPI)

    fig, ax = plt.subplots(figsize=(7, 5))
    _errbar(ax, rs_burst, "RS(15,11,2), GF(2⁴)", "o")
    _errbar(ax, bch_burst, "BCH(15,7,2)", "s")
    _setup(ax, "p_burst (вероятность пакета на кадр)", "FER",
           "Направление Б: RS vs BCH в пакетном канале (длина ≤ 8 бит), N=10⁵")
    fig.tight_layout()
    fig.savefig(RESULTS / "direction_b_burst_fer.png", dpi=DPI)

    ztest_table(rs_bsc, bch_bsc, "RS", "BCH", RESULTS / "direction_b_bsc_ztest.md")
    ztest_table(rs_burst, bch_burst, "RS", "BCH", RESULTS / "direction_b_burst_ztest.md")


def validation_plot():
    rows = load_csv(RESULTS / "validation_rs_symbol.csv")
    n, t = 15, 2

    def theory(ps):
        return 1.0 - sum(
            math.comb(n, i) * ps**i * (1 - ps) ** (n - i) for i in range(t + 1)
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    _errbar(ax, rows, "Монте-Карло, RS(15,11,2)", "o")
    grid = [rows[0]["p"] * (rows[-1]["p"] / rows[0]["p"]) ** (i / 99) for i in range(100)]
    ax.plot(grid, [theory(ps) for ps in grid], "k--",
            label="теория: P(>2 символьных ошибок из 15)")
    _setup(ax, "p_s (вероятность символьной ошибки)", "FER",
           "Валидация Монте-Карло: символьный канал, N=10⁵")
    fig.tight_layout()
    fig.savefig(RESULTS / "validation_symbol_channel.png", dpi=DPI)


if __name__ == "__main__":
    direction_a()
    direction_b()
    validation_plot()
    print("Готово: графики и таблицы в results/")
