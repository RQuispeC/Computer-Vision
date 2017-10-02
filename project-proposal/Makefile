all:
	pdflatex proposal
	pdflatex proposal
	
bib:
	pdflatex proposal
	bibtex proposal
	pdflatex proposal

clean: 
	rm -f *.aux *.bbl *.log *.blg *.dvi *.toc *.out

cleanall: clean
	rm -f proposal.pdf
