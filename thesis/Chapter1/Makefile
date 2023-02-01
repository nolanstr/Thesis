# Linux Makefile to create a final pdf of the project

# Variables
CHAPTER =    Chapter1
OVERVIEW =   0_Overview
BACKGROUND = 1_Background
AIMS =       2_Aims
IMPACT =     3_Impact

# PDF LaTeX specific
TEX = "pdflatex -interaction=nonstopmode -synctex=1 --shell-escape"

# You want latexmk to *always* run, because make does not have all the info.
# Also, include non-file targets in .PHONY so they are run regardless of any
# file of the given name existing.
.PHONY: ${CHAPTER}.pdf all clean

# The first rule in a Makefile is the one executed by default ("make"). It
# should always be the "all" rule, so that "make" and "make all" are identical.
all: ${CHAPTER}.pdf

# CUSTOM BUILD RULES

# In case you didn't know, '$@' is a variable holding the name of the target,
# and '$<' is a variable holding the (first) dependency of a rule.
# "raw2tex" and "dat2tex" are just placeholders for whatever custom steps
# you might have.

%.tex: %.raw
	./raw2tex $< > $@

%.tex: %.dat
	./dat2tex $< > $@
	
chapter: ${CHAPTER}.pdf
overview: ${OVERVIEW}.pdf
background: ${BACKGROUND}.pdf
aims: ${AIMS}.pdf
impact: ${IMPACT}.pdf

# MAIN LATEXMK RULE

# -pdf tells latexmk to generate PDF directly (instead of DVI).
# -pdflatex="" tells latexmk to call a specific backend with specific options.
# -use-make tells latexmk to call make for generating missing files.

# -interaction=nonstopmode keeps the pdflatex backend from stopping at a
# missing file reference and interactively asking you for an alternative.

# --shell-escape allows for *minted to run code highlighting

# -f forces latexmk to run until compiling has been complete regardless of
#  cross referencing (i.e. it continues to run until references are in the
#  correct location)

${CHAPTER}.pdf: ${CHAPTER}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${CHAPTER}.tex

${OVERVIEW}.pdf: ${OVERVIEW}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${OVERVIEW}.tex

${BACKGROUND}.pdf: ${BACKGROUND}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${BACKGROUND}.tex
	
${AIMS}.pdf: ${AIMS}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${AIMS}.tex
	
${IMPACT}.pdf: ${IMPACT}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${IMPACT}.tex

	
# Clean up unnecessary files
clean:
	latexmk -C

clear:	
	rm -rf auto *_minted-* *.log *.aux *.synctex.gz *.out *.toc *.run *.bcf *.lof *.lot *.tdo *.run.xml *.pdf *.bbl *.blg *.swp

release:
	# rm -rf Release # Once the folder is created, no need to create a new one
	# mkdir Release # Once the folder is created, no need to create a new one
	cp *.pdf Release
	make clear
	make clean