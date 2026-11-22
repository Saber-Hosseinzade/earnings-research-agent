#!/usr/bin/env bash
# One-time setup per account and region: turns on CloudWatch Transaction Search,
# which AgentCore Runtime needs to deliver traces. Not in CDK because it changes
# an account-wide setting that outlives any stack. Safe to run again.
# Undo: aws xray update-trace-segment-destination --destination XRay
set -euo pipefail

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGION="${AWS_REGION:?set AWS_REGION}"

# Lets X-Ray write spans to CloudWatch Logs, only on behalf of this account.
aws logs put-resource-policy --policy-name TransactionSearchAccess --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [{
    \"Sid\": \"TransactionSearchXRayAccess\",
    \"Effect\": \"Allow\",
    \"Principal\": {\"Service\": \"xray.amazonaws.com\"},
    \"Action\": \"logs:PutLogEvents\",
    \"Resource\": [
      \"arn:aws:logs:$REGION:$ACCOUNT:log-group:aws/spans:*\",
      \"arn:aws:logs:$REGION:$ACCOUNT:log-group:/aws/application-signals/data:*\"
    ],
    \"Condition\": {
      \"ArnLike\": {\"aws:SourceArn\": \"arn:aws:xray:$REGION:$ACCOUNT:*\"},
      \"StringEquals\": {\"aws:SourceAccount\": \"$ACCOUNT\"}
    }
  }]
}" > /dev/null

# Switching to the destination that is already set is an error, so only switch when needed.
if [ "$(aws xray get-trace-segment-destination --query Destination --output text)" != "CloudWatchLogs" ]; then
  aws xray update-trace-segment-destination --destination CloudWatchLogs > /dev/null
fi

# Spans hold full prompts and answers. These two groups belong to the account, not to a stack, so their
# retention is set here. Applied after the destination switch, which creates them if they don't exist yet.
SPAN_RETENTION_DAYS="${SPAN_RETENTION_DAYS:-7}"
for group in aws/spans /aws/application-signals/data; do
  if aws logs describe-log-groups --log-group-name-prefix "$group" --query "logGroups[?logGroupName=='$group']" --output text | grep -q .; then
    aws logs put-retention-policy --log-group-name "$group" --retention-in-days "$SPAN_RETENTION_DAYS"
  else
    echo "note: $group does not exist yet; run this script again after the first traced invocation" >&2
  fi
done
aws xray get-trace-segment-destination
