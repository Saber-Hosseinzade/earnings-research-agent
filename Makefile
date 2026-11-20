# Shortcuts for common commands. Run `make help` to list them.
# Target names use a hyphen: make treats ":" in a target name as a rule separator.

PROFILE ?= sandbox

.PHONY: help sso-login sso-whoami test kb-docs kb-sync corpus-terms ui ui-dev
UI_DEV_USER ?= local-viewer
UI_DEV_ROLE ?= viewer

help:  ## List the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-20s %s\n", $$1, $$2}'

sso-login:  ## Log in to AWS SSO (sandbox profile, device code)
	aws sso login --profile $(PROFILE) --use-device-code

sso-whoami:  ## Show the identity the sandbox profile resolves to
	aws sts get-caller-identity --profile $(PROFILE)

test:  ## Run the offline tests
	python -m pytest -q

kb-docs:  ## Build Knowledge Base passages from corpus/v2/raw into build/kb_docs (offline)
	python scripts/build_kb_docs.py

kb-sync: kb-docs  ## Upload passages to the KB bucket and run an ingestion job
	AWS_PROFILE=$(PROFILE) python scripts/sync_kb.py

corpus-terms:  ## Rebuild corpus/terms.json (words that are not tickers) from corpus/v2/raw (offline)
	python scripts/corpus_terms.py

ui:  ## Run the UI locally with Cognito sign-in (needs AGENT_RUNTIME_ARN; once: python scripts/ui_auth_config.py)
	cd ui && uv sync --frozen -q && AWS_PROFILE=$(PROFILE) AWS_REGION=eu-central-1 .venv/bin/streamlit run app.py

ui-dev:  ## Run the UI locally without sign-in, as UI_DEV_USER / UI_DEV_ROLE=viewer|reviewer (needs AGENT_RUNTIME_ARN)
	cd ui && uv sync --frozen -q && AWS_PROFILE=$(PROFILE) AWS_REGION=eu-central-1 UI_AUTH=dev \
	  UI_DEV_USER=$(UI_DEV_USER) UI_DEV_ROLE=$(UI_DEV_ROLE) .venv/bin/streamlit run app.py
