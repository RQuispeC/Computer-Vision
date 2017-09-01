all:
	pdflatex report
	pdflatex report
	
bib:
	pdflatex report
	bibtex report
	pdflatex report

clean: 
	rm -f *.aux *.bbl *.log *.blg *.dvi *.toc *.out

cleanall: clean
	rm -f report.pdf
