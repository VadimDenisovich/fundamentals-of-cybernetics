"""Программная проверка письменных примеров из раздела 1 отчёта.

Независимая (без galois) реализация арифметики GF(2^3), кодирования
и декодирования Берлекэмпа--Месси. Печатает все промежуточные
значения, которые приведены в отчёте вручную, и в конце сверяет
кодовые слова с библиотекой galois.

Запуск: python tools/verify_manual_examples.py
"""

import numpy as np
import galois

# --- GF(2^3), p(x) = x^3 + x + 1 ----------------------------------------
# Элементы как целые: биты (b2 b1 b0) = b2*x^2 + b1*x + b0.
EXP = [1]  # EXP[i] = alpha^i
for _ in range(6):
    v = EXP[-1] << 1
    if v & 0b1000:
        v ^= 0b1011  # x^3 = x + 1
    EXP.append(v)
LOG = {EXP[i]: i for i in range(7)}


def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[(LOG[a] + LOG[b]) % 7]


def gf_div(a, b):
    if a == 0:
        return 0
    return EXP[(LOG[a] - LOG[b]) % 7]


def gf_pow_alpha(i):
    return EXP[i % 7]


def fmt(v):
    if v == 0:
        return "0"
    if v == 1:
        return "1"
    return f"a^{LOG[v]}" if LOG[v] > 1 else "a"


def poly_str(coeffs):
    """coeffs: старшая степень первой."""
    terms = []
    deg = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        d = deg - i
        cs = "" if (c == 1 and d > 0) else fmt(c)
        if d == 0:
            terms.append(cs or "1")
        elif d == 1:
            terms.append(f"{cs}x" if cs else "x")
        else:
            terms.append(f"{cs}x^{d}" if cs else f"x^{d}")
    return " + ".join(terms) if terms else "0"


def poly_mul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            out[i + j] ^= gf_mul(ca, cb)
    return out


def poly_mod(num, den):
    num = list(num)
    while len(num) >= len(den) and any(num):
        if num[0] == 0:
            num.pop(0)
            continue
        coef = gf_div(num[0], den[0])
        for i in range(len(den)):
            num[i] ^= gf_mul(coef, den[i])
        num.pop(0)
    pad = len(den) - 1 - len(num)
    return [0] * pad + num


def poly_eval(coeffs, x):
    acc = 0
    for c in coeffs:
        acc = gf_mul(acc, x) ^ c
    return acc


def berlekamp_massey(syndromes, t):
    """BM по Lin & Costello; печатает таблицу итераций.

    Возвращает Lambda(x) как список коэффициентов от младшей степени:
    [1, l1, l2, ...].
    """
    lam = [1]          # Lambda(x)
    b = [1]            # B(x)
    L = 0
    print("\n  Таблица итераций Берлекэмпа--Месси:")
    print("  k | S_k  | d (дискрепанс) | Lambda(x)            | L | B(x)")
    for k in range(1, 2 * t + 1):
        s = syndromes[k - 1]
        d = s
        for i in range(1, L + 1):
            if i < len(lam):
                d ^= gf_mul(lam[i], syndromes[k - 1 - i])
        if d == 0:
            b = [0] + b
        else:
            t_poly = list(lam)
            xb = [0] + b
            new_lam = [0] * max(len(lam), len(xb))
            for i in range(len(new_lam)):
                v1 = lam[i] if i < len(lam) else 0
                v2 = gf_mul(d, xb[i]) if i < len(xb) else 0
                new_lam[i] = v1 ^ v2
            if 2 * L <= k - 1:
                b = [gf_div(c, d) for c in t_poly]
                L = k - L
            else:
                b = [0] + b
            lam = new_lam
        lam_str = poly_str(list(reversed(lam)))
        b_str = poly_str(list(reversed(b)))
        print(f"  {k} | {fmt(s):4} | {fmt(d):14} | {lam_str:20} | {L} | {b_str}")
    return lam


def chien_forney(lam, syndromes, n, t):
    """Поиск Чиеня + формула Форни (narrow-sense, c=1).
    Возвращает [(позиция_степени, значение_ошибки)]."""
    # Корни Lambda: перебор x = a^-j  <=>  позиция ошибки j
    error_positions = []
    print("\n  Поиск Чиеня (подстановка x = a^{-j}):")
    for j in range(n):
        x = gf_pow_alpha(-j)
        val = 0
        for i, c in enumerate(lam):
            val ^= gf_mul(c, gf_pow_alpha((LOG[x] * i) % 7) if x != 0 else 0) if c else 0
        # прямое вычисление надёжнее:
        val = 0
        for i, c in enumerate(lam):
            term = c
            for _ in range(i):
                term = gf_mul(term, x)
            val ^= term
        mark = " <-- корень, ошибка в позиции x^%d" % j if val == 0 else ""
        print(f"    j={j}: Lambda(a^-{j}) = Lambda({fmt(x)}) = {fmt(val)}{mark}")
        if val == 0:
            error_positions.append(j)

    # Omega(x) = S(x) * Lambda(x) mod x^{2t}
    s_poly = list(syndromes)  # S1 + S2 x + ... (младшая степень первой)
    omega_full = [0] * (len(s_poly) + len(lam) - 1)
    for i, cs in enumerate(s_poly):
        for k, cl in enumerate(lam):
            omega_full[i + k] ^= gf_mul(cs, cl)
    omega = omega_full[: 2 * t]
    print(f"\n  Omega(x) = S(x)*Lambda(x) mod x^{2*t} = "
          f"{poly_str(list(reversed(omega)))}")

    # Lambda'(x): формальная производная (в GF(2) выживают нечётные степени)
    lam_deriv = [lam[i] if i % 2 == 1 else 0 for i in range(1, len(lam))]
    print(f"  Lambda'(x) = {poly_str(list(reversed(lam_deriv)))}")

    errors = []
    for j in error_positions:
        x_inv = gf_pow_alpha(-j)
        om = 0
        for i, c in enumerate(omega):
            term = c
            for _ in range(i):
                term = gf_mul(term, x_inv)
            om ^= term
        ld = 0
        for i, c in enumerate(lam_deriv):
            term = c
            for _ in range(i):
                term = gf_mul(term, x_inv)
            ld ^= term
        e = gf_div(om, ld)
        print(f"  Форни: e_{j} = Omega({fmt(x_inv)})/Lambda'({fmt(x_inv)}) "
              f"= {fmt(om)}/{fmt(ld)} = {fmt(e)}")
        errors.append((j, e))
    return errors


def rs_example():
    print("=" * 72)
    print("ПРИМЕР 1: RS(7,3), t=2 над GF(2^3), p(x) = x^3+x+1")
    print("=" * 72)

    print("\nТаблица GF(8):")
    for i in range(7):
        print(f"  a^{i} = {EXP[i]:03b} = {EXP[i]}")

    # g(x) = (x-a)(x-a^2)(x-a^3)(x-a^4)
    g = [1]
    for i in range(1, 5):
        g = poly_mul(g, [1, gf_pow_alpha(i)])
    print(f"\ng(x) = (x+a)(x+a^2)(x+a^3)(x+a^4) = {poly_str(g)}")

    # Сообщение m = (a, 1, a^3)
    m = [2, 1, 3]
    print(f"\nСообщение m = ({', '.join(fmt(v) for v in m)}) = {m}")
    print(f"m(x) = {poly_str(m)}")

    shifted = m + [0, 0, 0, 0]
    parity = poly_mod(shifted, g)
    print(f"m(x)*x^4 = {poly_str(shifted)}")
    print(f"r(x) = m(x)*x^4 mod g(x) = {poly_str(parity)}")

    codeword = m + parity
    print(f"c(x) = m(x)*x^4 + r(x) = {poly_str(codeword)}")
    print(f"c = ({', '.join(fmt(v) for v in codeword)}) = {codeword}")

    for i in range(1, 5):
        v = poly_eval(codeword, gf_pow_alpha(i))
        print(f"  проверка: c(a^{i}) = {fmt(v)}")
        assert v == 0

    # Ошибки: e(x) = a^2*x^5 + a*x^2  (позиции 5 и 2)
    e_deg = {5: 4, 2: 2}  # значение a^2=4 в x^5, a=2 в x^2
    received = list(codeword)
    n = 7
    for deg, val in e_deg.items():
        received[n - 1 - deg] ^= val
    print(f"\nОшибка e(x) = a^2 x^5 + a x^2")
    print(f"Принято r = ({', '.join(fmt(v) for v in received)}) = {received}")

    syndromes = []
    for i in range(1, 5):
        s = poly_eval(received, gf_pow_alpha(i))
        syndromes.append(s)
        print(f"  S_{i} = r(a^{i}) = {fmt(s)}")

    lam = berlekamp_massey(syndromes, t=2)
    print(f"\n  Lambda(x) = {poly_str(list(reversed(lam)))}")

    errors = chien_forney(lam, syndromes, n=7, t=2)
    corrected = list(received)
    for j, e in errors:
        corrected[n - 1 - j] ^= e
    print(f"\nИсправлено: c' = ({', '.join(fmt(v) for v in corrected)}) = {corrected}")
    assert corrected == codeword, "исправленное слово != исходному!"
    print("OK: исправленное слово совпало с переданным.")

    rs = galois.ReedSolomon(7, 3)
    lib = list(np.asarray(rs.encode(rs.field([m])))[0])
    assert lib == codeword, f"galois: {lib} != ручному {codeword}"
    dec = np.asarray(rs.decode(rs.field([received])))[0]
    assert list(dec) == m
    print(f"OK: galois кодирует m в то же слово {lib} и декодирует r в m.")


def bch_example():
    print("\n" + "=" * 72)
    print("ПРИМЕР 2: BCH(7,4), t=1 (код Хэмминга), g(x) = x^3+x+1")
    print("=" * 72)

    g = [1, 0, 1, 1]  # x^3 + x + 1 над GF(2)
    m = [1, 0, 0, 1]
    print(f"\nСообщение m = {m},  m(x) = x^3 + 1")
    shifted = m + [0, 0, 0]
    parity = poly_mod(shifted, g)
    print(f"m(x)*x^3 = {poly_str(shifted)}")
    print(f"r(x) = m(x)*x^3 mod g(x) = {poly_str(parity)}")
    codeword = m + parity
    print(f"c = {codeword}")

    n = 7
    err_deg = 4
    received = list(codeword)
    received[n - 1 - err_deg] ^= 1
    print(f"\nОшибка в позиции x^{err_deg}: r = {received}")

    s1 = poly_eval(received, gf_pow_alpha(1))
    s2 = poly_eval(received, gf_pow_alpha(2))
    print(f"S_1 = r(a) = {fmt(s1)}")
    print(f"S_2 = r(a^2) = {fmt(s2)} (= S_1^2 = {fmt(gf_mul(s1, s1))})")
    assert s2 == gf_mul(s1, s1)

    lam = berlekamp_massey([s1, s2], t=1)
    print(f"\n  Lambda(x) = {poly_str(list(reversed(lam)))}")
    assert lam == [1, s1]

    # Корень: x = a^-j => 1 + S1 * a^-j = 0 => a^j = S1
    j = LOG[s1]
    print(f"  Корень Lambda: a^j = S_1 = {fmt(s1)} => позиция ошибки j = {j}")
    assert j == err_deg

    corrected = list(received)
    corrected[n - 1 - j] ^= 1
    print(f"Исправлено: {corrected}")
    assert corrected == codeword
    print("OK: исправленное слово совпало с переданным.")

    bch = galois.BCH(7, 4)
    lib = list(np.asarray(bch.encode(galois.GF2([m])))[0])
    assert lib == codeword, f"galois: {lib} != ручному {codeword}"
    dec = np.asarray(bch.decode(galois.GF2([received])))[0]
    assert list(dec) == m
    print(f"OK: galois кодирует m в то же слово {lib} и декодирует r в m.")


if __name__ == "__main__":
    rs_example()
    bch_example()
    print("\nВсе письменные примеры подтверждены.")
