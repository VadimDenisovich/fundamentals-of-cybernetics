"""Метрики и статистика: доверительные интервалы и z-тест для долей."""

import math


def wald_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95%-й доверительный интервал Вальда: p +- z * sqrt(p(1-p)/N)."""
    if n == 0:
        return 0.0, 1.0
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n)
    return max(0.0, p_hat - half), min(1.0, p_hat + half)


def two_proportion_ztest(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Двухвыборочный z-тест для долей (pooled).

    H0: p1 == p2. Возвращает (z, p_value, двусторонний).
    """
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p_value
