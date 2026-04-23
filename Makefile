PYTHON ?= ./.venv/bin/python
PIP ?= ./.venv/bin/pip
PYTEST ?= ./.venv/bin/pytest
RUFF ?= ./.venv/bin/ruff

AUTOQEC_IDEATOR_BACKEND ?= codex-cli
AUTOQEC_IDEATOR_MODEL   ?= gpt-5.4
AUTOQEC_CODER_BACKEND   ?= codex-cli
AUTOQEC_CODER_MODEL     ?= gpt-5.4-codex
AUTOQEC_ANALYST_BACKEND ?= claude-cli
AUTOQEC_ANALYST_MODEL   ?= claude-haiku-4-5

COMMON_ENV = \
	AUTOQEC_IDEATOR_BACKEND=$(AUTOQEC_IDEATOR_BACKEND) \
	AUTOQEC_IDEATOR_MODEL=$(AUTOQEC_IDEATOR_MODEL) \
	AUTOQEC_CODER_BACKEND=$(AUTOQEC_CODER_BACKEND) \
	AUTOQEC_CODER_MODEL=$(AUTOQEC_CODER_MODEL) \
	AUTOQEC_ANALYST_BACKEND=$(AUTOQEC_ANALYST_BACKEND) \
	AUTOQEC_ANALYST_MODEL=$(AUTOQEC_ANALYST_MODEL)

ENV ?= autoqec/envs/builtin/surface_d5_depol.yaml
ROUNDS ?= 10
PROFILE ?= dev

.PHONY: install test test-integration coverage lint run run-nollm verify demo-2 run-all-claude run-cheap build-trap-fixtures

install:
	$(PIP) install -e '.[dev]'

test:
	$(PYTEST) tests/ -m "not integration" -v

coverage:
	$(PYTEST) --cov

test-integration:
	$(PYTEST) tests/ -m "integration" -v --run-integration

lint:
	$(RUFF) check autoqec cli tests scripts

run:
	$(COMMON_ENV) $(PYTHON) -m cli.autoqec run $(ENV) --rounds $(ROUNDS) --profile $(PROFILE)

run-nollm:
	$(PYTHON) -m cli.autoqec run $(ENV) --rounds $(ROUNDS) --profile $(PROFILE) --no-llm

verify:
	@echo "verify CLI is owned by the verification slice and is not implemented in this branch"
	@exit 1

demo-2:
	bash demos/demo-2-bb72/run.sh

run-all-claude:
	$(MAKE) run AUTOQEC_IDEATOR_BACKEND=claude-cli AUTOQEC_CODER_BACKEND=claude-cli

run-cheap:
	$(MAKE) run AUTOQEC_IDEATOR_MODEL=claude-haiku-4-5

build-trap-fixtures:
	$(PYTHON) scripts/build_trap_fixtures.py --env autoqec/envs/builtin/surface_d5_depol.yaml
