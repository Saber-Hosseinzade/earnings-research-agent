#!/usr/bin/env python3
"""Upload the passages to the Knowledge Base bucket and run an ingestion job.

    python scripts/build_kb_docs.py
    AWS_PROFILE=sandbox python scripts/sync_kb.py            # reads IDs from the EarningsResearchAgentKB stack

Uploads every file in build/kb_docs, deletes bucket objects that are no longer
there, starts an ingestion job, waits for it and prints its statistics. Record
them (date, vector store, counts) in the notes: they are the sync record.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "build" / "kb_docs"
STACK = "EarningsResearchAgentKB"
REGION = "eu-central-1"


def stack_outputs(cfn, stack: str) -> dict[str, str]:
    outs = cfn.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", [])
    return {o["OutputKey"]: o["OutputValue"] for o in outs}


def upload(s3, bucket: str, docs: Path) -> tuple[int, int]:
    local = {f.name: f for f in docs.iterdir() if f.is_file()}
    remote = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        remote.update(o["Key"] for o in page.get("Contents", []))
    for name, f in sorted(local.items()):
        s3.upload_file(str(f), bucket, name)
    stale = sorted(remote - local.keys())
    for i in range(0, len(stale), 1000):
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in stale[i : i + 1000]]})
    return len(local), len(stale)


def ingest(agent, kb_id: str, ds_id: str, poll_s: int = 10) -> dict:
    job = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)["ingestionJob"]
    while job["status"] in ("STARTING", "IN_PROGRESS", "STOPPING"):
        time.sleep(poll_s)
        job = agent.get_ingestion_job(
            knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job["ingestionJobId"]
        )["ingestionJob"]
        print(f"  {job['status']}", flush=True)
    return job


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docs", type=Path, default=DOCS)
    ap.add_argument("--stack", default=STACK)
    ap.add_argument("--region", default=REGION)
    args = ap.parse_args(argv)

    if not args.docs.is_dir() or not any(args.docs.iterdir()):
        print(f"{args.docs} is empty: run scripts/build_kb_docs.py first", file=sys.stderr)
        return 2

    session = boto3.Session(region_name=args.region)
    out = stack_outputs(session.client("cloudformation"), args.stack)
    files, deleted = upload(session.client("s3"), out["DocsBucketName"], args.docs)
    print(f"uploaded {files} files, deleted {deleted} stale objects -> s3://{out['DocsBucketName']}")

    job = ingest(session.client("bedrock-agent"), out["KnowledgeBaseId"], out["DataSourceId"])
    print(f"vector store: {out.get('VectorStore')}  job: {job['ingestionJobId']}  status: {job['status']}")
    for k, v in job.get("statistics", {}).items():
        print(f"  {k}: {v}")
    for reason in job.get("failureReasons", []):
        print(f"  failure: {reason}")
    return 0 if job["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
