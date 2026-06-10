"""Модели каналов с ошибками (самостоятельная реализация).

Все каналы работают с кадрами битов: вход и выход — массив numpy
формы (n_frames, frame_bits) со значениями {0, 1} (dtype uint8).
Источник случайности (numpy Generator) передаётся снаружи, что
обеспечивает воспроизводимость при фиксированном seed.
"""

import numpy as np


def bits_to_symbols(bits: np.ndarray, m: int) -> np.ndarray:
    """Упаковка битов в m-битные символы, старший бит первым.

    (n_frames, n_sym * m) бит -> (n_frames, n_sym) символов.
    """
    n_frames, n_bits = bits.shape
    if n_bits % m != 0:
        raise ValueError(f"число бит {n_bits} не кратно размеру символа {m}")
    weights = (1 << np.arange(m - 1, -1, -1)).astype(np.int64)
    return (bits.reshape(n_frames, -1, m).astype(np.int64) * weights).sum(axis=2)


def symbols_to_bits(symbols: np.ndarray, m: int) -> np.ndarray:
    """Распаковка m-битных символов в биты, старший бит первым."""
    n_frames, n_sym = symbols.shape
    shifts = np.arange(m - 1, -1, -1)
    bits = (symbols[:, :, None].astype(np.int64) >> shifts) & 1
    return bits.reshape(n_frames, n_sym * m).astype(np.uint8)


class Channel:
    """Базовый класс канала."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def transmit(self, bits: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BSCChannel(Channel):
    """Двоичный симметричный канал (BSC).

    Каждый бит независимо инвертируется с вероятностью p:
        y_i = x_i XOR e_i,  e_i ~ Bernoulli(p).
    """

    def __init__(self, p: float, rng: np.random.Generator):
        super().__init__(rng)
        if not 0.0 <= p <= 1.0:
            raise ValueError("p должно лежать в [0, 1]")
        self.p = p

    def transmit(self, bits: np.ndarray) -> np.ndarray:
        flips = (self.rng.random(bits.shape) < self.p).astype(np.uint8)
        return bits ^ flips


class SymbolErrorChannel(Channel):
    """Канал символьных ошибок.

    Кадр делится на m-битные символы; каждый символ независимо
    с вероятностью p_s заменяется на случайный ДРУГОЙ символ
    (XOR с равномерной ненулевой маской из GF(2^m) \\ {0}).
    """

    def __init__(self, p_s: float, m: int, rng: np.random.Generator):
        super().__init__(rng)
        if not 0.0 <= p_s <= 1.0:
            raise ValueError("p_s должно лежать в [0, 1]")
        self.p_s = p_s
        self.m = m

    def transmit(self, bits: np.ndarray) -> np.ndarray:
        symbols = bits_to_symbols(bits, self.m)
        hit = self.rng.random(symbols.shape) < self.p_s
        masks = self.rng.integers(1, 2**self.m, size=symbols.shape)
        corrupted = np.where(hit, symbols ^ masks, symbols)
        return symbols_to_bits(corrupted, self.m)


class BurstChannel(Channel):
    """Пакетный канал.

    С вероятностью p_burst на кадр вносится один пакет ошибок:
    непрерывный участок случайной длины L ~ U{1, ..., max_len}
    со случайным началом s ~ U{0, ..., frame_bits - L}; все биты
    участка инвертируются. Пакет целиком лежит внутри кадра.
    """

    def __init__(self, p_burst: float, max_len: int, rng: np.random.Generator):
        super().__init__(rng)
        if not 0.0 <= p_burst <= 1.0:
            raise ValueError("p_burst должно лежать в [0, 1]")
        if max_len < 1:
            raise ValueError("max_len должно быть >= 1")
        self.p_burst = p_burst
        self.max_len = max_len

    def transmit(self, bits: np.ndarray) -> np.ndarray:
        n_frames, n_bits = bits.shape
        if self.max_len > n_bits:
            raise ValueError("длина пакета больше длины кадра")
        # Все случайные величины тянутся для каждого кадра независимо от hit,
        # чтобы поток rng не зависел от исходов (воспроизводимость).
        hit = self.rng.random(n_frames) < self.p_burst
        lengths = self.rng.integers(1, self.max_len + 1, size=n_frames)
        starts = self.rng.integers(0, n_bits - lengths + 1)
        positions = np.arange(n_bits)
        burst_mask = (
            (positions >= starts[:, None])
            & (positions < (starts + lengths)[:, None])
            & hit[:, None]
        )
        return bits ^ burst_mask.astype(np.uint8)
