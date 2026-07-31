# Study Lab — root Makefile.
# Builds/runs the Java packages (Data_structures, Algorithms, Problems) and the
# Jupyter notebooks under Machine_learning/.
#
#   make setup                    -> verify the toolchain (javac, python3, conda,
#                                    xelatex) — errors out naming whatever
#                                    is missing and can't be auto-installed —
#                                    then create/update the `ml` conda env from
#                                    Machine_learning/environment.yml and register
#                                    its Jupyter kernel. TeX Live is installed
#                                    automatically via apt-get if absent.
#   make run F=Stack              -> runs Data_structures.Stack_Main
#   make run F=Merge_sort         -> runs Algorithms.Merge_sort_Main
#   make run F=Magical_cows      -> runs Problems.Magical_cows_Main
#   make run-nb N=<name|path>     -> executes a notebook top-to-bottom, in place
#                                    (N=islp_lab_ch03 finds Machine_learning/**/islp_lab_ch03.ipynb)
#   make run-latest               -> runs the most recently modified source file:
#                                    .java -> its *_Main, .ipynb -> execute the
#                                    notebook, Machine_learning .py -> run it in the ml env
#   make run-all                  -> runs every *_Main AND executes every notebook
#                                    under Machine_learning/, logs everything to
#                                    test-results.log, prints only failures
#                                    (notebooks are skipped with a notice when
#                                    neither conda nor jupyter is installed)
#   make notes-pdf                -> compiles every .tex under Notes/ (recursively)
#                                    into a same-named .pdf with xelatex, run twice
#                                    so the table of contents resolves. Build one
#                                    document only: make notes-pdf N=Stats
#   make notes-clean              -> delete the LaTeX aux files under Notes/
#                                    (.aux .log .toc .out .fls ...), keeping the PDFs
#   make list                     -> list runnable classes + notebooks, newest first
#   make clean                    -> delete .class files, .ipynb_checkpoints, and
#                                    GENERATED artifacts under Machine_learning/
#                                    (anything not a protected type, outside
#                                    Machine_learning/data/ — see below)
#   make clean-all                -> clean + strip all output cells from every
#                                    notebook under Machine_learning/ (commit-ready
#                                    notebooks) + delete every generated PDF and
#                                    LaTeX aux file under Notes/
#
# clean's protected types under Machine_learning/ (never deleted): .ipynb .py
# .yml .yaml .md, everything inside Machine_learning/data/, and hidden files.
# Everything else there (pngs, pdfs, html exports, csv dumps outside data/, ...)
# is treated as generated output and removed. Add extensions to ML_KEEP to protect more.
#
# Notebook execution/stripping uses the `ml` conda env when conda is available
# (conda run -n ml), otherwise whatever `jupyter` is on PATH.

DS_DIR   := Data_structures
ALG_DIR  := Algorithms
PROB_DIR := Problems
DIRS     := $(DS_DIR) $(ALG_DIR) $(PROB_DIR)
SRCS     := $(foreach d,$(DIRS),$(wildcard $(d)/*.java))

ML_DIR   := Machine_learning
ML_KEEP  := ipynb py yml yaml md
NBS       = $(shell find $(ML_DIR) -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' 2>/dev/null)

NOTES_DIR := Notes
# Each Notes/*.tex is a standalone document (article, 10pt, single column) with
# its full preamble inlined — page geometry and font size live in the .tex, not
# here. XeLaTeX is required: the notes use \newunicodechar and system fonts.
TEX       := xelatex
TEXFLAGS  := -interaction=nonstopmode -halt-on-error
TEX_AUX   := aux log toc out fls fdb_latexmk synctex.gz nav snm vrb bbl blg

# Run jupyter inside the ml conda env when conda exists; fall back to PATH.
JUPYTER  := $(shell command -v conda >/dev/null 2>&1 && echo "conda run -n ml jupyter" || echo "jupyter")
MLPY     := $(shell command -v conda >/dev/null 2>&1 && echo "conda run -n ml python" || echo "python3")

.PHONY: setup run run-nb run-latest run-all notes-pdf notes-clean list clean clean-all


# One-time toolchain + environment setup. Fails fast, naming what's missing.
# Also ensures pandoc + xelatex are present (auto-installed via apt-get if not).
setup:
	@ok=1; \
	if command -v javac >/dev/null 2>&1; then \
	  echo "  javac  : $$(javac --version)"; \
	else echo "ERROR: javac not found — install a JDK (macOS: brew install openjdk; container: apt-get install -y default-jdk-headless)"; ok=0; fi; \
	if command -v python3 >/dev/null 2>&1; then \
	  echo "  python : $$(python3 --version)"; \
	else echo "ERROR: python3 not found — install Python 3 (macOS: comes with Miniforge below)"; ok=0; fi; \
	if command -v conda >/dev/null 2>&1; then \
	  echo "  conda  : $$(conda --version)"; \
	else echo "ERROR: conda not found — install Miniforge (see Conda_setup_mac.md), then re-run 'make setup'"; ok=0; fi; \
	if [ $$ok -eq 0 ]; then exit 1; fi; \
	if [ ! -f $(ML_DIR)/environment.yml ]; then \
	  echo "ERROR: $(ML_DIR)/environment.yml not found"; exit 1; fi; \
	if conda env list | grep -qE '^ml[[:space:]]'; then \
	  echo "conda env 'ml' exists — updating from $(ML_DIR)/environment.yml"; \
	  conda env update -n ml -f $(ML_DIR)/environment.yml --prune; \
	else \
	  echo "creating conda env 'ml' from $(ML_DIR)/environment.yml"; \
	  conda env create -f $(ML_DIR)/environment.yml; \
	fi; \
	conda run -n ml python -m ipykernel install --user --name ml --display-name "Python (ml)"; \
	echo "--- checking latex-to-pdf toolchain (XeLaTeX + fonts) ---"; \
	if command -v xelatex >/dev/null 2>&1; then \
	  echo "  xelatex: $$(xelatex --version | head -1)"; \
	else \
	  echo "xelatex not found — installing TeX Live"; \
	  sudo apt-get update && sudo apt-get install -y \
	    texlive-xetex texlive-latex-recommended texlive-latex-extra \
	    texlive-fonts-recommended fonts-dejavu-core \
	    || { echo "ERROR: failed to install TeX Live"; exit 1; }; \
	fi; \
	missing=""; \
	for sty in fontspec newunicodechar titlesec fancyhdr enumitem microtype \
	           geometry hyperref longtable booktabs fancyvrb framed; do \
	  kpsewhich $$sty.sty >/dev/null 2>&1 || missing="$$missing $$sty"; \
	done; \
	if [ -n "$$missing" ]; then \
	  echo "  missing LaTeX packages:$$missing"; \
	  echo "  installing (sudo apt-get install -y texlive-latex-recommended texlive-latex-extra)"; \
	  sudo apt-get update && sudo apt-get install -y \
	    texlive-latex-recommended texlive-latex-extra \
	    || { echo "ERROR: failed to install the missing LaTeX packages"; exit 1; }; \
	else echo "  latex pkgs: all present"; fi; \
	if fc-list 2>/dev/null | grep -qi 'DejaVu Sans Mono'; then \
	  echo "  fonts  : DejaVu Sans Mono found"; \
	else \
	  echo "DejaVu Sans Mono not found (the notes use it for code) — installing"; \
	  sudo apt-get update && sudo apt-get install -y fonts-dejavu-core \
	    || { echo "ERROR: failed to install fonts-dejavu-core"; exit 1; }; \
	fi; \
	echo "setup complete — 'conda activate ml' to work interactively"

# Build everything, run <F>_Main from whichever package contains <F>.java, then clean.
run:
	@if [ -z "$(F)" ]; then echo "usage: make run F=ClassName"; exit 1; fi
	@javac -d . $(SRCS)
	@FQN=""; \
	for d in $(DIRS); do \
	  if [ -f "$$d/$(F).java" ]; then FQN="$$d.$(F)_Main"; break; fi; \
	done; \
	if [ -z "$$FQN" ]; then echo "class '$(F)' not found in: $(DIRS)"; $(MAKE) -s clean; exit 1; fi; \
	echo "running $$FQN"; echo; \
	java $$FQN; status=$$?; \
	$(MAKE) -s clean; \
	exit $$status

# Execute a notebook top-to-bottom, saving outputs in place.
# N can be a path (Machine_learning/trees/islp_lab_ch08.ipynb) or a bare name.
run-nb:
	@if [ -z "$(N)" ]; then echo "usage: make run-nb N=<notebook name or path>"; exit 1; fi
	@nb="$(N)"; \
	if [ ! -f "$$nb" ]; then \
	  nb=$$(find $(ML_DIR) -name "$(N).ipynb" -not -path '*/.ipynb_checkpoints/*' | head -1); \
	fi; \
	if [ -z "$$nb" ] || [ ! -f "$$nb" ]; then echo "notebook '$(N)' not found under $(ML_DIR)/"; exit 1; fi; \
	echo "executing $$nb"; \
	$(JUPYTER) nbconvert --to notebook --execute --inplace "$$nb"

# Run the most recently modified source file across the whole lab:
#   .java under the packages -> its *_Main; .ipynb under Machine_learning/ ->
#   execute it; .py under Machine_learning/ -> run it in the ml env.
run-latest:
	@latest=$$(ls -t $(foreach d,$(DIRS),$(d)/*.java) $$(find $(ML_DIR) \( -name '*.ipynb' -o -name '*.py' \) -not -path '*/.ipynb_checkpoints/*' 2>/dev/null) 2>/dev/null | head -1); \
	if [ -z "$$latest" ]; then echo "no runnable sources found"; exit 1; fi; \
	echo "latest: $$latest"; \
	case "$$latest" in \
	  *.ipynb) $(MAKE) run-nb N="$$latest" ;; \
	  *.py)    echo; $(MLPY) "$$latest" ;; \
	  *.java)  $(MAKE) run F=$$(basename $$latest .java) ;; \
	esac

# Run every Java *_Main and execute every notebook; log full output, print only failures.
run-all:
	@javac -d . $(SRCS)
	@: > test-results.log
	@fail=0; \
	for f in $(SRCS); do \
	  dir=$$(dirname $$f); pkg=$$(basename $$dir); base=$$(basename $$f .java); \
	  if [ -f "$$dir/$${base}_Main.class" ]; then \
	    main="$$pkg.$${base}_Main"; \
	    echo "=== $$main ===" >> test-results.log; \
	    if java $$main >> test-results.log 2>&1; then \
	      echo "" >> test-results.log; \
	    else \
	      echo "FAILED: $$main"; fail=1; \
	      echo "" >> test-results.log; \
	    fi; \
	  fi; \
	done; \
	nbs="$(NBS)"; \
	if [ -n "$$nbs" ]; then \
	  if command -v conda >/dev/null 2>&1 || command -v jupyter >/dev/null 2>&1; then \
	    for nb in $$nbs; do \
	      echo "=== $$nb ===" >> test-results.log; \
	      if $(JUPYTER) nbconvert --to notebook --execute --inplace "$$nb" >> test-results.log 2>&1; then \
	        echo "" >> test-results.log; \
	      else \
	        echo "FAILED: $$nb"; fail=1; \
	        echo "" >> test-results.log; \
	      fi; \
	    done; \
	  else \
	    echo "skipping notebooks (neither conda nor jupyter found)"; \
	    echo "=== notebooks skipped: no conda/jupyter ===" >> test-results.log; \
	  fi; \
	fi; \
	$(MAKE) -s clean; \
	if [ $$fail -eq 0 ]; then echo "all runnables passed (see test-results.log)"; \
	else echo "some runnables failed (see test-results.log)"; fi

# Compile every LaTeX file under Notes/ (recursively) into a same-named PDF.
# Each .tex is standalone; xelatex runs twice so \tableofcontents resolves.
# Build a single document with:  make notes-pdf N=Stats
notes-pdf:
	@if [ ! -d $(NOTES_DIR) ]; then echo "no $(NOTES_DIR)/ directory found"; exit 0; fi
	@command -v $(TEX) >/dev/null 2>&1 || { echo "$(TEX) not found — run 'make setup' first"; exit 1; }
	@if [ -n "$(N)" ]; then \
	  texs=$$(find $(NOTES_DIR) -name "$(N).tex"); \
	  [ -n "$$texs" ] || { echo "no $(N).tex found under $(NOTES_DIR)/"; exit 1; }; \
	else \
	  texs=$$(find $(NOTES_DIR) -name '*.tex' | sort); \
	fi; \
	if [ -z "$$texs" ]; then echo "no .tex files found under $(NOTES_DIR)/"; exit 0; fi; \
	fail=0; \
	for f in $$texs; do \
	  dir=$$(dirname "$$f"); base=$$(basename "$$f" .tex); \
	  echo "compiling $$f -> $$dir/$$base.pdf"; \
	  ( cd "$$dir" && \
	    $(TEX) $(TEXFLAGS) "$$base.tex" >"$$base.build.log" 2>&1 && \
	    $(TEX) $(TEXFLAGS) "$$base.tex" >>"$$base.build.log" 2>&1 ) || { \
	      echo "  FAILED: $$f — first errors:"; \
	      grep -m5 -A3 '^!' "$$dir/$$base.build.log" | sed 's/^/    /'; \
	      echo "    (full log: $$dir/$$base.build.log)"; \
	      fail=1; continue; }; \
	  rm -f "$$dir/$$base.build.log"; \
	done; \
	$(MAKE) -s notes-clean; \
	if [ $$fail -eq 0 ]; then echo "all notes compiled"; \
	else echo "some notes failed to compile"; exit 1; fi

# Delete LaTeX intermediates under Notes/, keeping the .tex sources and the PDFs.
notes-clean:
	@if [ -d $(NOTES_DIR) ]; then \
	  find $(NOTES_DIR) -type f \( $(foreach e,$(TEX_AUX),-name '*.$(e)' -o) -false \) \
	    -delete 2>/dev/null; \
	fi

# List runnable classes and notebooks, newest first.
list:
	@ls -t $(foreach d,$(DIRS),$(d)/*.java) | sed 's|.*/||;s|\.java||' | awk '{print "  make run F="$$1}'
	@find $(ML_DIR) -name '*.ipynb' -not -path '*/.ipynb_checkpoints/*' 2>/dev/null | \
	  xargs -I{} echo "  make run-nb N={}"

# clean: .class files everywhere; under Machine_learning/: checkpoint dirs and
# any GENERATED artifact — a non-hidden file whose extension is not in ML_KEEP,
# outside Machine_learning/data/.
clean:
	@find . -name '*.class' -delete
	@if [ -d $(ML_DIR) ]; then \
	  find $(ML_DIR) -type d -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null; \
	  find $(ML_DIR) -type f \
	    -not -path '$(ML_DIR)/data/*' \
	    -not -name '.*' \
	    $(foreach e,$(ML_KEEP),-not -name '*.$(e)') \
	    -print -delete | sed 's/^/  cleaned: /'; \
	fi
	rm -rf ./Machine_learning/logs
	rm -rf ./Machine_learning/lightning_logs

# clean-all: clean + strip every notebook's output cells (small, diff-friendly
# commits) + remove every generated PDF and LaTeX aux file under Notes/.
clean-all: clean notes-clean
	@if [ -n "$(NBS)" ]; then \
	  echo "stripping notebook outputs:"; \
	  for nb in $(NBS); do echo "  $$nb"; done; \
	  $(JUPYTER) nbconvert --clear-output --inplace $(NBS); \
	else echo "no notebooks to strip"; fi
	@if [ -d $(NOTES_DIR) ]; then \
	  pdfs=$$(find $(NOTES_DIR) -name '*.pdf'); \
	  if [ -n "$$pdfs" ]; then \
	    echo "removing generated notes PDFs:"; \
	    for p in $$pdfs; do echo "  $$p"; rm -f "$$p"; done; \
	  else echo "no notes PDFs to remove"; fi; \
	fi