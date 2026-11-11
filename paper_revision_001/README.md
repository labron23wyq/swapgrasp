# Paper revision 001

English two-page draft for the language-driven grasp detection assignment. It reports the three-seed matched experiment and keeps the later two-seed optimization follow-up separate because it uses a different split and is not an independent confirmation set.

Compile on a TeX Live host from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

The author is Yuquan Wang and the source repository is https://github.com/labron23wyq/swapgrasp. Before submission, rename the final PDF to `FirstName_LastName.pdf` as specified in the assignment.
