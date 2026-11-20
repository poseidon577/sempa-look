# SEMPA-LAB entry points. `make -n <target>` prints the commands without running them.
#
#   make test          full pytest suite (unit tests download MiniLM on first use)
#   make check-docs    fast README / docs / config consistency checks, no model download
#   make repro         every Stage 4 paper experiment (MPNet, about 5 h on 4 CPUs)
#   make channel       every channel-aware experiment (many hours)
#   make figures       all figures and the paper comparison table from committed results
#
# Override PYTHON to use another interpreter, and CONFIGS / CHANNEL_CONFIGS to run a subset:
#   make repro CONFIGS=configs/repro/coco_k8.yaml

PYTHON ?= python
CONFIGS ?= $(sort $(wildcard configs/repro/*.yaml))
CHANNEL_CONFIGS ?= $(sort $(wildcard configs/channel/*.yaml))

.PHONY: help test check-docs repro channel figures

help:
	@sed -n '2,10p' Makefile

test:
	$(PYTHON) -m pytest -q

check-docs:
	$(PYTHON) -m pytest -q tests/test_repo_docs.py

repro:
	@for cfg in $(CONFIGS); do \
		echo "== $$cfg"; \
		$(PYTHON) -m sempalab.repro --config $$cfg || exit 1; \
	done

channel:
	@for cfg in $(CHANNEL_CONFIGS); do \
		echo "== $$cfg"; \
		$(PYTHON) -m sempalab.channel_experiment --config $$cfg || exit 1; \
	done

figures:
	$(PYTHON) scripts/make_figures.py
	$(PYTHON) scripts/make_channel_figures.py
	$(PYTHON) scripts/compare_paper.py
