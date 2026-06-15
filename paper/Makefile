MAIN   = paper
LATEX  = pdflatex
BIBTEX = bibtex
OUT    = build

.PHONY: all clean

all:
	mkdir -p $(OUT)
	$(LATEX)  -output-directory=$(OUT) -interaction=nonstopmode $(MAIN).tex
	$(BIBTEX) $(OUT)/$(MAIN)
	$(LATEX)  -output-directory=$(OUT) -interaction=nonstopmode $(MAIN).tex
	$(LATEX)  -output-directory=$(OUT) -interaction=nonstopmode $(MAIN).tex
	cp $(OUT)/$(MAIN).pdf .

clean:
	rm -rf $(OUT) *.pdf
