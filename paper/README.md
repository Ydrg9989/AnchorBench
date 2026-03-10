# Template

Template and style files for CoLM 2026

## Building the PDF

```bash
cd paper
./compile_pdf.sh
```

If you see **"I can't find the format file \`pdflatex.fmt'!"**, your conda TeX Live is broken (missing `TeXLive::TLUtils.pm`). Either:

- **Use system TeX:** Install TeX Live from your OS (e.g. `sudo apt install texlive-latex-extra` or `sudo dnf install texlive-scheme-medium`) and run `./compile_pdf.sh` again—it will prefer `/usr/bin/pdflatex` when present.
- **Or run without conda in PATH:** `env PATH=/usr/bin:/usr/local/bin:$PATH latexmk -pdf colm2026_conference.tex`
