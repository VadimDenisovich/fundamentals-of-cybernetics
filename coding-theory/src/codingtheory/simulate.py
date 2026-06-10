"""Цикл Монте-Карло: сообщение -> кодер -> канал -> декодер -> метрики."""

import csv
import time

import numpy as np

from .channels import bits_to_symbols, symbols_to_bits
from .metrics import wald_ci

CSV_FIELDS = [
    "code",
    "channel",
    "p",
    "n_blocks",
    "frame_errors",
    "fer",
    "fer_ci_lo",
    "fer_ci_hi",
    "bit_errors",
    "total_info_bits",
    "ber",
    "ber_ci_lo",
    "ber_ci_hi",
    "decode_time_total_s",
    "decode_time_per_block_ms",
    "throughput_kbit_s",
]


def simulate_point(
    codec,
    make_channel,
    p: float,
    n_blocks: int,
    seed: int,
    batch_size: int = 20_000,
    channel_label: str = "",
) -> dict:
    """Одна точка кривой: n_blocks кадров при параметре канала p.

    make_channel(p, rng) -> Channel. Один и тот же seed даёт идентичные
    сообщения и идентичные реализации шума для любых кодеков с одинаковыми
    (n, k, symbol_bits) — это позволяет честно сравнивать библиотеки.
    """
    rng = np.random.default_rng(seed)
    channel = make_channel(p, rng)

    frame_errors = 0
    bit_errors = 0
    decode_time = 0.0

    done = 0
    while done < n_blocks:
        nb = min(batch_size, n_blocks - done)
        msgs = codec.random_messages(nb, rng)
        codewords = codec.encode(msgs)
        tx_bits = symbols_to_bits(codewords, codec.symbol_bits)
        rx_bits = channel.transmit(tx_bits)
        rx = bits_to_symbols(rx_bits, codec.symbol_bits)

        t0 = time.perf_counter()
        decoded = codec.decode(rx)
        decode_time += time.perf_counter() - t0

        frame_errors += int((decoded != msgs).any(axis=1).sum())
        dec_bits = symbols_to_bits(decoded, codec.symbol_bits)
        msg_bits = symbols_to_bits(msgs, codec.symbol_bits)
        bit_errors += int((dec_bits != msg_bits).sum())
        done += nb

    total_info_bits = n_blocks * codec.info_bits
    fer = frame_errors / n_blocks
    ber = bit_errors / total_info_bits
    fer_lo, fer_hi = wald_ci(fer, n_blocks)
    ber_lo, ber_hi = wald_ci(ber, total_info_bits)

    return {
        "code": codec.name,
        "channel": channel_label,
        "p": p,
        "n_blocks": n_blocks,
        "frame_errors": frame_errors,
        "fer": fer,
        "fer_ci_lo": fer_lo,
        "fer_ci_hi": fer_hi,
        "bit_errors": bit_errors,
        "total_info_bits": total_info_bits,
        "ber": ber,
        "ber_ci_lo": ber_lo,
        "ber_ci_hi": ber_hi,
        "decode_time_total_s": decode_time,
        "decode_time_per_block_ms": 1000.0 * decode_time / n_blocks,
        "throughput_kbit_s": total_info_bits / decode_time / 1000.0 if decode_time > 0 else float("inf"),
    }


def run_sweep(
    codec,
    make_channel,
    p_range,
    n_blocks: int,
    seed: int,
    batch_size: int = 20_000,
    channel_label: str = "",
    verbose: bool = True,
) -> list[dict]:
    """Прогон по сетке параметров канала. Перед замерами — прогрев
    (JIT-компиляция galois не должна попадать в тайминги)."""
    warm_rng = np.random.default_rng(0)
    codec.decode(codec.encode(codec.random_messages(2, warm_rng)))

    rows = []
    for p in p_range:
        row = simulate_point(
            codec, make_channel, float(p), n_blocks, seed,
            batch_size=batch_size, channel_label=channel_label,
        )
        rows.append(row)
        if verbose:
            print(
                f"[{codec.name} | {channel_label}] p={p:.5g}  "
                f"FER={row['fer']:.3e}  BER={row['ber']:.3e}  "
                f"t_dec={row['decode_time_per_block_ms']:.3f} ms/блок",
                flush=True,
            )
    return rows


def save_csv(rows: list[dict], path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path) -> list[dict]:
    with open(path, newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            for key in row:
                if key not in ("code", "channel"):
                    row[key] = float(row[key])
            rows.append(row)
        return rows
