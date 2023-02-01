# Linux Makefile to create a final pdf of the project

# Variables
CHAPTER =     Chapter5
SUMMARY =     0_Summary_of_key_findings
SUMMARY2 =    1_Chapter2_summary
SUMMARY3 =    2_Chapter3_summary
SUMMARY4 =    3_Chapter4_summary
CONCLUSION =  4_Conclusion
FUTUREWORK =  5_Future_Work

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
summary: ${SUMMARY}.pdf
summary2: ${SUMMARY2}.pdf
summary3: ${SUMMARY3}.pdf
summary4: ${SUMMARY4}.pdf
conclusion: ${CONCLUSION}.pdf
limitations: ${LIMITATIONS}.pdf
futureWork: ${FUTUREWORK}.pdf


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

${SUMMARY}.pdf: ${SUMMARY}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${SUMMARY}.tex

${SUMMARY2}.pdf: ${SUMMARY2}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${SUMMARY2}.tex

${SUMMARY3}.pdf: ${SUMMARY3}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${SUMMARY3}.tex

${SUMMARY4}.pdf: ${SUMMARY4}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${SUMMARY4}.tex

${CONCLUSION}.pdf: ${CONCLUSION}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${CONCLUSION}.tex

${LIMITATIONS}.pdf: ${LIMITATIONS}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${LIMITATIONS}.tex

${FUTUREWORK}.pdf: ${FUTUREWORK}.tex
	latexmk -pdf -pdflatex=${TEX} -f -use-make ${FUTUREWORK}.tex

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
