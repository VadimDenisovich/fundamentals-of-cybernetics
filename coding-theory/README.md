# Сравнительный анализ помехоустойчивости кодов Рида–Соломона и BCH

Лабораторная работа: сравнение реализаций RS-кода в библиотеках `galois` и `reedsolo`
(Направление А) и сравнение кодов RS(15,11,2) / BCH(15,7,2) в каналах BSC и Burst
(Направление Б) методом Монте-Карло.

**Отчёт: [REPORT.md](REPORT.md) · [REPORT.pdf](REPORT.pdf)** (формулы в LaTeX, отображаются и на GitHub, и в PDF)

## Структура

```
src/codingtheory/        # пакет
  channels.py            #   модели каналов: BSC, SymbolError, Burst (своя реализация)
  codecs.py              #   обёртки кодеков: galois RS/BCH, reedsolo RS
  metrics.py             #   CI Вальда, двухвыборочный z-тест
  simulate.py            #   цикл Монте-Карло, CSV
  validation.py          #   валидация на 5 известных векторах
experiments/
  run_direction_a.py     # эксперимент А: galois vs reedsolo, RS(255,223), BSC, N=10⁴
  run_direction_b.py     # эксперимент Б: RS(15,11) vs BCH(15,7), BSC/Burst, N=10⁵
  plot_results.py        # графики и статистические таблицы
tools/
  verify_manual_examples.py  # проверка письменных примеров из отчёта (GF(2³), БМ)
report/
  build.sh               # сборка REPORT.pdf (pandoc + lualatex, шрифт CMU Serif)
  header.tex             # оформление PDF
tests/                   # pytest
results/                 # CSV, PNG, md-таблицы
```

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest

pytest                                    # тесты
python -m codingtheory.validation         # валидация на известных векторах
python tools/verify_manual_examples.py    # проверка письменных примеров отчёта

python experiments/run_direction_a.py     # ~1.5 ч
python experiments/run_direction_b.py     # ~1 ч
python experiments/plot_results.py        # графики из CSV

bash report/build.sh                      # пересборка REPORT.pdf (нужны pandoc + lualatex)
```

Все эксперименты детерминированы (фиксированные seed) и воспроизводимы.
