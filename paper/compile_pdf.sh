#!/usr/bin/env bash
# Compile colm2026_conference.tex to PDF.
# Use system TeX if available (conda's TeX Live often has broken mktexfmt).
set -e
cd "$(dirname "$0")"

# Prefer system pdflatex/latexmk over conda's broken TeX
if command -v /usr/bin/pdflatex &>/dev/null; then
  export PATH="/usr/bin:/usr/local/bin:$PATH"
fi

if command -v latexmk &>/dev/null; then
  latexmk -pdf -interaction=nonstopmode colm2026_conference.tex
else
  pdflatex -interaction=nonstopmode colm2026_conference.tex
  bibtex colm2026_conference
  pdflatex -interaction=nonstopmode colm2026_conference.tex
  pdflatex -interaction=nonstopmode colm2026_conference.tex
fi

echo "Done: colm2026_conference.pdf"
