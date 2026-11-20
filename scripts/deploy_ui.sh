#!/usr/bin/env bash
# Deploy the hosted UI (infra/ui_stack.py) for a demo window. The runtime must be deployed first.
#
#   scripts/build_agent.sh && (cd infra && npx cdk deploy EarningsResearchAgentRuntime --exclusively)
#   scripts/deploy_ui.sh
#   cd infra && npx cdk destroy EarningsResearchAgentUi --exclusively     # after the demo, before the runtime
#
# Steps: auth + image stacks; the Cognito client secret into Secrets Manager; the image built by CodeBuild
# unless its tag (the hash of ui/) is already in ECR; the UI stack. On the first deploy the endpoint is not
# known yet, so the script then adds it to Cognito's callback URLs and the container and deploys both again.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
export AWS_PROFILE=${AWS_PROFILE:-sandbox} AWS_REGION=eu-central-1
cd "$ROOT/infra"

out() { aws cloudformation describe-stacks --stack-name "$1" \
          --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text; }

[ -n "$(ls -A "$ROOT/build/agent" 2>/dev/null)" ] || { echo "build/agent is empty: run scripts/build_agent.sh" >&2; exit 1; }
aws cloudformation describe-stacks --stack-name EarningsResearchAgentRuntime >/dev/null 2>&1 \
  || { echo "EarningsResearchAgentRuntime is not deployed: deploy it first" >&2; exit 1; }

npx cdk deploy EarningsResearchAgentAuth EarningsResearchAgentUiImage --exclusively --require-approval never
python "$ROOT/scripts/ui_auth_config.py" --to-secrets-manager

TAG=$(out EarningsResearchAgentUiImage ImageTag)
REPO=$(out EarningsResearchAgentUiImage RepositoryName)
if aws ecr describe-images --repository-name "$REPO" --image-ids imageTag="$TAG" >/dev/null 2>&1; then
  echo "image $TAG already built"
else
  BUILD=$(aws codebuild start-build --project-name "$(out EarningsResearchAgentUiImage ProjectName)" \
            --query build.id --output text)
  echo "building image $TAG ($BUILD)"
  while STATUS=$(aws codebuild batch-get-builds --ids "$BUILD" --query "builds[0].buildStatus" --output text) \
        && [ "$STATUS" = IN_PROGRESS ]; do sleep 15; done
  [ "$STATUS" = SUCCEEDED ] || { echo "image build $STATUS: see log group /earnings-research-agent/ui-build" >&2; exit 1; }
fi

npx cdk deploy EarningsResearchAgentUi --exclusively --require-approval never
HOST=$(out EarningsResearchAgentUi Endpoint | sed -E 's#^https?://##; s#/.*$##')
if [ "$HOST" != "$(cat .ui_hostname 2>/dev/null || true)" ]; then
  echo "$HOST" > .ui_hostname
  echo "new endpoint: adding it to Cognito and the container"
  npx cdk deploy EarningsResearchAgentAuth EarningsResearchAgentUi --exclusively --require-approval never
fi
echo "UI: https://$HOST"
