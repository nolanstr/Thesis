# Linux Makefile to create a final pdf of the project

# Variables
CHAPTER =      Chapter2
ABSTRACT =     0_Abstract
INTRODUCTION = 1_Introduction
METHODS =      2_Methods
RESULTS =      3_Results
DISCUSSION =   4_Discussion
CONCLUSION =   5_Conclusion
ACKNOWLEDGE =  6_Acknowledgment

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
abstract: ${ABSTRACT}.pdf
introduction: ${INTRODUCTION}.pdf
methods: ${METHODS}.pdf
results: ${RESULTS}.pdf
discussion: ${DISCUSSION}.pdf
conclusion: ${CONCLUSION}.pdf
acknowledge: ${ACKNOWLEDGE}.pdf

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

${ABSTRACT}.pdf: ${ABSTRACT}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${ABSTRACT}.tex

${INTRODUCTION}.pdf: ${INTRODUCTION}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${INTRODUCTION}.tex
	
${METHODS}.pdf: ${METHODS}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${METHODS}.tex
	
${RESULTS}.pdf: ${RESULTS}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${RESULTS}.tex
	
${DISCUSSION}.pdf: ${DISCUSSION}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${DISCUSSION}.tex
	
${CONCLUSION}.pdf: ${CONCLUSION}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${CONCLUSION}.tex

${ACKNOWLEDGE}.pdf: ${ACKNOWLEDGE}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${ACKNOWLEDGE}.tex
	
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