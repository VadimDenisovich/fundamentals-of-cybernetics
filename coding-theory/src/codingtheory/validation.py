"""Валидация на известных векторах (требование задания: 3--5 векторов).

Запуск: python -m codingtheory.validation
"""

import galois
import numpy as np

from .channels import BSCChannel, BurstChannel, SymbolErrorChannel
from .codecs import GaloisRS, ReedsoloRS


def check_rs7_manual_vector() -> tuple[bool, str]:
    """Вектор 1. RS(7,3) над GF(2^3): кодовое слово, вычисленное вручную
    в отчёте (раздел 1), должно совпадать с кодером galois.

    m = (a, 1, a^3) = (2, 1, 3);  c = (a, 1, a^3, a^4, 0, a^2, a^6)
    = (2, 1, 3, 6, 0, 4, 5) при p(x) = x^3 + x + 1.
    """
    rs = galois.ReedSolomon(7, 3)
    message = [2, 1, 3]
    expected = [2, 1, 3, 6, 0, 4, 5]
    actual = list(np.asarray(rs.encode(rs.field([message])))[0])
    return actual == expected, f"ожидалось {expected}, получено {actual}"


def check_bch7_manual_vector() -> tuple[bool, str]:
    """Вектор 2. BCH(7,4) (эквивалент кода Хэмминга, g(x) = x^3 + x + 1):
    кодовое слово из ручного примера отчёта.

    m = (1,0,0,1);  c = (1,0,0,1,1,1,0).
    """
    bch = galois.BCH(7, 4)
    g_expected = galois.Poly.Int(0b1011)  # x^3 + x + 1
    if bch.generator_poly != g_expected:
        return False, f"g(x) = {bch.generator_poly}, ожидалось {g_expected}"
    message = [1, 0, 0, 1]
    expected = [1, 0, 0, 1, 1, 1, 0]
    actual = list(np.asarray(bch.encode(galois.GF2([message])))[0])
    return actual == expected, f"ожидалось {expected}, получено {actual}"


def check_bch15_generator_poly() -> tuple[bool, str]:
    """Вектор 3. BCH(15,7): табличный порождающий многочлен
    g(x) = x^8 + x^7 + x^6 + x^4 + 1 (Lin & Costello, Table 6.1)."""
    bch = galois.BCH(15, 7)
    g_expected = galois.Poly.Int(0b111010001)
    ok = bch.generator_poly == g_expected
    return ok, f"g(x) = {bch.generator_poly}, ожидалось {g_expected}"


def check_rs255_cross_library() -> tuple[bool, str]:
    """Вектор 4. RS(255,223): galois и reedsolo (fcr=1) дают побайтово
    идентичные кодовые слова на одном сообщении -- реализации одного кода."""
    rng = np.random.default_rng(12345)
    msgs = rng.integers(0, 256, size=(5, 223))
    cw_galois = GaloisRS(255, 223).encode(msgs)
    cw_reedsolo = ReedsoloRS(255, 223, fcr=1).encode(msgs)
    ok = bool((cw_galois == cw_reedsolo).all())
    diff = int((cw_galois != cw_reedsolo).sum())
    return ok, f"несовпадающих байт: {diff} из {cw_galois.size}"


def check_channels_statistics() -> tuple[bool, str]:
    """Вектор 5. Статистическая валидация каналов: эмпирические частоты
    ошибок согласуются с заданными параметрами (допуск 4 сигмы)."""
    rng = np.random.default_rng(777)
    n_frames, n_bits = 2000, 600
    zeros = np.zeros((n_frames, n_bits), dtype=np.uint8)
    msgs_total = n_frames * n_bits
    details = []
    ok = True

    p = 0.05
    flipped = int(BSCChannel(p, rng).transmit(zeros).sum())
    sigma = np.sqrt(p * (1 - p) * msgs_total)
    ok_bsc = abs(flipped - p * msgs_total) < 4 * sigma
    ok &= ok_bsc
    details.append(f"BSC: {flipped}/{msgs_total} инверсий при p={p} ({'ok' if ok_bsc else 'FAIL'})")

    p_s, m = 0.1, 4
    out = SymbolErrorChannel(p_s, m, rng).transmit(zeros)
    n_sym = n_frames * n_bits // m
    sym_err = int((out.reshape(n_frames, -1, m).sum(axis=2) > 0).sum())
    sigma = np.sqrt(p_s * (1 - p_s) * n_sym)
    ok_sym = abs(sym_err - p_s * n_sym) < 4 * sigma
    ok &= ok_sym
    details.append(f"SymbolError: {sym_err}/{n_sym} символьных ошибок при p_s={p_s} ({'ok' if ok_sym else 'FAIL'})")

    p_b, max_len = 0.3, 8
    out = BurstChannel(p_b, max_len, rng).transmit(zeros)
    frames_hit = int((out.sum(axis=1) > 0).sum())
    sigma = np.sqrt(p_b * (1 - p_b) * n_frames)
    ok_burst = abs(frames_hit - p_b * n_frames) < 4 * sigma
    # пакет непрерывен и не длиннее max_len
    for frame in out[out.sum(axis=1) > 0][:50]:
        idx = np.flatnonzero(frame)
        ok_burst &= len(idx) <= max_len and (idx[-1] - idx[0] + 1) == len(idx)
    ok &= ok_burst
    details.append(f"Burst: {frames_hit}/{n_frames} кадров с пакетом при p_burst={p_b} ({'ok' if ok_burst else 'FAIL'})")

    return bool(ok), "; ".join(details)


ALL_CHECKS = [
    ("RS(7,3) GF(8): ручной пример == galois", check_rs7_manual_vector),
    ("BCH(7,4): ручной пример == galois", check_bch7_manual_vector),
    ("BCH(15,7): табличный g(x)", check_bch15_generator_poly),
    ("RS(255,223): galois == reedsolo", check_rs255_cross_library),
    ("Каналы: статистическая валидация", check_channels_statistics),
]


def main() -> int:
    failed = 0
    for name, check in ALL_CHECKS:
        ok, details = check()
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {details}")
        failed += not ok
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
