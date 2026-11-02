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

aws xray update-trace-segment-destination --destination CloudWatchLogs > /dev/null
aws xray get-trace-segment-destination
