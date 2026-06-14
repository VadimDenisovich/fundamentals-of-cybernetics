#!/usr/bin/env bash
# Сборка PDF из REPORT.md: pandoc + lualatex, шрифт CMU Serif (полная кириллица + математика).
# Требования: pandoc, lualatex (TeX Live / MacTeX), шрифт CMU Serif (пакет cm-unicode).
#
# Запуск из каталога coding-theory:   bash report/build.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # -> coding-theory/

pandoc REPORT.md \
  -o REPORT.pdf \
  --pdf-engine=lualatex \
  --toc --toc-depth=2 \
  -V documentclass=scrartcl \
  -V classoption=11pt \
  -V classoption=titlepage \
  -V geometry:a4paper,margin=2.2cm \
  -V mainfont="CMU Serif" \
  -V sansfont="CMU Sans Serif" \
  -V monofont="CMU Typewriter Text" \
  -V mathfont="Latin Modern Math" \
  -V colorlinks=true \
  -V linkcolor=accent \
  -V urlcolor=accentlight \
  -V toccolor=accent \
  --include-in-header=report/header.tex \
  --resource-path=.:report

echo "Готово: REPORT.pdf"
