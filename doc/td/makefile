.SUFFIXES: .tex .aux .bib

filenames=td_1



pspdf:	ps 
	$(foreach var,$(filenames),ps2pdf $(var).ps;)

pdf: 
	pdflatex ${filename}
	bibtex ${filename} ||true
	pdflatex ${filename}
	pdflatex ${filename}

pdf-print: ps
	ps2pdf -dColorConversionStrategy=/LeaveColorUnchanged -dPDFSETTINGS=/printer ${filename}.ps

text: html
	html2text -width 100 -style pretty ${filename}/${filename}.html | sed -n '/./,$$p' | head -n-2 >${filename}.txt

html:
	@#latex2html -split +0 -info "" -no_navigation ${filename}
	htlatex ${filename}

ps:	dvi
	$(foreach var,$(filenames),dvips -t letter $(var).dvi;)

dvi:
	$(foreach var,$(filenames),latex $(var);)
	$(foreach var,$(filenames),bibtex $(var)||true;)
	$(foreach var,$(filenames),latex $(var);)
	$(foreach var,$(filenames),latex $(var);)

read:  pdf
	xpdf ${filename}.pdf 

aread: pdf
	acroread ${filename}.pdf

gvread: ps
	gv ${filename}.ps 

clean:
	rm -f *.ps *.pdf *.log *.aux *.out *.dvi *.bbl *.blg *.mtc* *.toc *.maf
