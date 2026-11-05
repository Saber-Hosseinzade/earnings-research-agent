# Shortcuts for common commands. Run `make help` to list them.
# Target names use a hyphen: make treats ":" in a target name as a rule separator.

PROFILE ?= sandbox

.PHONY: help sso-login sso-whoami test

help:  ## List the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-20s %s\n", $$1, $$2}'

sso-login:  ## Log in to AWS SSO (sandbox profile, device code)
	aws sso login --profile $(PROFILE) --use-device-code

sso-whoami:  ## Show the identity the sandbox profile resolves to
	aws sts get-caller-identity --profile $(PROFILE)

test:  ## Run the offline tests
	python -m pytest -q
