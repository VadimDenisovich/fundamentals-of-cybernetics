"""Обёртки над библиотечными кодеками с единым интерфейсом.

Сообщения и кодовые слова — массивы символов формы (n_frames, k) и
(n_frames, n); для двоичного BCH символ = бит. Все коды систематические:
кодовое слово = [сообщение | проверочные символы].
"""

import numpy as np
import galois
import reedsolo


class Codec:
    """Единый интерфейс кодека."""

    name: str
    n: int
    k: int
    t: int
    symbol_bits: int

    @property
    def frame_bits(self) -> int:
        return self.n * self.symbol_bits

    @property
    def info_bits(self) -> int:
        return self.k * self.symbol_bits

    def random_messages(self, n_frames: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2**self.symbol_bits, size=(n_frames, self.k))

    def encode(self, msgs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, received: np.ndarray) -> np.ndarray:
        """Возвращает декодированные сообщения (n_frames, k).

        При отказе декодера (обнаружена неисправимая комбинация) кадру
        присваивается информационная часть принятого слова — стандартная
        стратегия "passthrough", одинаковая для всех обёрток.
        """
        raise NotImplementedError


class GaloisRS(Codec):
    """Код Рида--Соломона из библиотеки galois."""

    def __init__(self, n: int, k: int):
        self.code = galois.ReedSolomon(n, k)
        self.name = f"galois RS({n},{k})"
        self.n, self.k, self.t = n, k, self.code.t
        self.symbol_bits = self.code.field.degree

    def encode(self, msgs: np.ndarray) -> np.ndarray:
        return np.asarray(self.code.encode(self.code.field(msgs)), dtype=np.int64)

    def decode(self, received: np.ndarray) -> np.ndarray:
        dec, n_errors = self.code.decode(self.code.field(received), errors=True)
        dec = np.asarray(dec, dtype=np.int64)
        fails = np.atleast_1d(n_errors) == -1
        if fails.any():
            dec[fails] = received[fails, : self.k]
        return dec


class GaloisBCH(Codec):
    """Двоичный код BCH из библиотеки galois."""

    def __init__(self, n: int, k: int):
        self.code = galois.BCH(n, k)
        self.name = f"galois BCH({n},{k})"
        self.n, self.k, self.t = n, k, self.code.t
        self.symbol_bits = 1

    def encode(self, msgs: np.ndarray) -> np.ndarray:
        return np.asarray(self.code.encode(galois.GF2(msgs)), dtype=np.int64)

    def decode(self, received: np.ndarray) -> np.ndarray:
        dec, n_errors = self.code.decode(galois.GF2(received), errors=True)
        dec = np.asarray(dec, dtype=np.int64)
        fails = np.atleast_1d(n_errors) == -1
        if fails.any():
            dec[fails] = received[fails, : self.k]
        return dec


class ReedsoloRS(Codec):
    """Код Рида--Соломона из библиотеки reedsolo (чистый Python).

    fcr=1 выравнивает порождающий многочлен с galois (narrow-sense,
    корни alpha^1..alpha^2t); оба используют примитивный многочлен 0x11D.
    """

    symbol_bits = 8

    def __init__(self, n: int, k: int, fcr: int = 1):
        if n != 255:
            raise ValueError("reedsolo используется для кодов над GF(2^8), n=255")
        self.codec = reedsolo.RSCodec(n - k, nsize=n, fcr=fcr)
        self.name = f"reedsolo RS({n},{k})"
        self.n, self.k, self.t = n, k, (n - k) // 2

    def encode(self, msgs: np.ndarray) -> np.ndarray:
        out = np.empty((len(msgs), self.n), dtype=np.int64)
        for i, msg in enumerate(msgs):
            cw = self.codec.encode(bytearray(msg.astype(np.uint8).tobytes()))
            out[i] = np.frombuffer(bytes(cw), dtype=np.uint8)
        return out

    def decode(self, received: np.ndarray) -> np.ndarray:
        out = np.empty((len(received), self.k), dtype=np.int64)
        for i, frame in enumerate(received):
            try:
                dec, _, _ = self.codec.decode(bytearray(frame.astype(np.uint8).tobytes()))
                out[i] = np.frombuffer(bytes(dec), dtype=np.uint8)
            except reedsolo.ReedSolomonError:
                out[i] = frame[: self.k]
        return out
