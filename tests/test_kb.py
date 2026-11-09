"""Knowledge Base, offline: the passage builder on the invented call, and the stack in both vector-store modes."""

import importlib.util
import json
import sys
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(__file__).resolve().parent / "fixtures" / "corpus"
sys.path.insert(0, str(ROOT / "infra"))

from kb_stack import EMBEDDING_MODEL_ID, EarningsKnowledgeBaseStack  # noqa: E402

spec = importlib.util.spec_from_file_location("build_kb_docs", ROOT / "scripts" / "build_kb_docs.py")
kb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kb)


# --- passage builder -------------------------------------------------------


def test_short_text_is_one_part():
    assert kb.split_text("One sentence. Two.", limit=100) == ["One sentence. Two."]


def test_long_text_splits_at_sentence_ends_and_keeps_every_word():
    text = " ".join(f"Sentence number {i} ends here." for i in range(40))
    parts = kb.split_text(text, limit=200)
    assert len(parts) > 1
    assert all(len(p) <= 200 for p in parts)
    assert all(p.endswith(".") for p in parts)
    assert " ".join(parts).split() == text.split()


def test_decimal_point_is_not_a_sentence_end():
    parts = kb.split_text("Revenue was $11.5 billion. " * 3, limit=40)
    assert all("$11.5 billion." in p for p in parts)


def test_build_writes_one_file_per_turn_with_metadata(tmp_path):
    counts = kb.build(CORPUS, tmp_path)
    call = json.loads((CORPUS / "DEMO_2026Q1.json").read_text())
    non_empty = sum(1 for t in call["data"] if (t.get("text") or "").strip())
    assert counts == {"DEMO_2026Q1": non_empty}

    meta = json.loads((tmp_path / "DEMO_2026Q1_t001.txt.metadata.json").read_text())["metadataAttributes"]
    turn = call["data"][1]
    assert meta["passage_id"] == "DEMO_2026Q1_t001"
    assert (meta["ticker"], meta["quarter"], meta["turn"]) == ("DEMO", "2026Q1", 1)
    assert meta["speaker"] == (turn.get("speakerName") or "Unknown")
    assert len(json.dumps(meta)) < 2048  # S3 Vectors filterable-metadata cap

    body = (tmp_path / "DEMO_2026Q1_t001.txt").read_text()
    assert body.startswith(f"{call['eventTitle']} | DEMO 2026Q1 |")
    assert turn["text"].strip() in body


def test_long_turn_becomes_numbered_parts_sharing_the_turn():
    call = {"eventTitle": "X", "fiscalYear": 2026, "fiscalQuarter": 1,
            "data": [{"speakerName": "A", "speakerRole": "CEO", "startSeconds": 10.5,
                      "text": "Word word word word. " * 300}]}
    ps = kb.passages_for_call(call, "X", "2026Q1")
    assert len(ps) > 1
    assert [p["id"] for p in ps][:2] == ["X_2026Q1_t000_p1", "X_2026Q1_t000_p2"]
    assert {p["metadata"]["turn"] for p in ps} == {0}
    assert ps[0]["metadata"]["start_seconds"] == 11  # rounded half up, as in eval/check_golden.py


def test_build_refuses_an_empty_corpus(tmp_path):
    with pytest.raises(FileNotFoundError):
        kb.build(tmp_path, tmp_path / "out")


# --- stack -------------------------------------------------------------------


def template(vector_store):
    app = cdk.App()
    return Template.from_stack(EarningsKnowledgeBaseStack(app, "KB", vector_store=vector_store))


def role_statements(t):
    roles = t.find_resources("AWS::IAM::Role", {"Properties": {"Description": Match.absent()}})
    kb_role = [r for r in roles.values()
               if "bedrock.amazonaws.com" in json.dumps(r["Properties"]["AssumeRolePolicyDocument"])]
    assert len(kb_role) == 1
    found = [s for p in kb_role[0]["Properties"].get("Policies", []) for s in p["PolicyDocument"]["Statement"]]
    for policy in t.find_resources("AWS::IAM::Policy").values():
        found += policy["Properties"]["PolicyDocument"]["Statement"]
    return kb_role[0], found


@pytest.mark.parametrize("vector_store", ["s3vectors", "opensearch"])
def test_kb_role_has_no_wildcard_resources_and_one_model(vector_store):
    role, found = role_statements(template(vector_store))
    assert all(s["Resource"] != "*" for s in found)
    invoke = [s for s in found if "bedrock:InvokeModel" in str(s["Action"])]
    assert len(invoke) == 1 and f"foundation-model/{EMBEDDING_MODEL_ID}" in json.dumps(invoke[0]["Resource"])
    trust = json.dumps(role["Properties"]["AssumeRolePolicyDocument"])
    assert "aws:SourceAccount" in trust and "knowledge-base/*" in trust


@pytest.mark.parametrize("vector_store", ["s3vectors", "opensearch"])
def test_data_source_does_not_chunk(vector_store):
    template(vector_store).has_resource_properties(
        "AWS::Bedrock::DataSource",
        {"VectorIngestionConfiguration": {"ChunkingConfiguration": {"ChunkingStrategy": "NONE"}}},
    )


def test_docs_bucket_is_private_and_tls_only():
    t = template("s3vectors")
    t.has_resource_properties("AWS::S3::Bucket", {"PublicAccessBlockConfiguration": {
        "BlockPublicAcls": True, "BlockPublicPolicy": True, "IgnorePublicAcls": True, "RestrictPublicBuckets": True}})
    assert "aws:SecureTransport" in json.dumps(t.find_resources("AWS::S3::BucketPolicy"))


def test_s3vectors_mode_keeps_passage_text_out_of_filterable_metadata():
    t = template("s3vectors")
    t.has_resource_properties("AWS::S3Vectors::Index", {
        "Dimension": 1024,
        "MetadataConfiguration": {"NonFilterableMetadataKeys": ["AMAZON_BEDROCK_TEXT", "AMAZON_BEDROCK_METADATA"]},
    })
    t.has_resource_properties("AWS::Bedrock::KnowledgeBase", {"StorageConfiguration": {"Type": "S3_VECTORS"}})
    t.resource_count_is("AWS::OpenSearchServerless::Collection", 0)
    _, found = role_statements(t)
    vec = [s for s in found if "s3vectors:" in str(s["Action"])]
    assert vec and all("VectorIndex" in json.dumps(s["Resource"]) for s in vec)


def test_opensearch_mode_is_a_small_vector_collection():
    t = template("opensearch")
    t.has_resource_properties("AWS::OpenSearchServerless::Collection",
                              {"Type": "VECTORSEARCH", "StandbyReplicas": "DISABLED"})
    t.has_resource_properties("AWS::Bedrock::KnowledgeBase", {"StorageConfiguration": {"Type": "OPENSEARCH_SERVERLESS"}})
    t.resource_count_is("AWS::S3Vectors::Index", 0)
    _, found = role_statements(t)
    aoss = [s for s in found if "aoss:" in str(s["Action"])]
    assert aoss and all("Collection" in json.dumps(s["Resource"]) for s in aoss)


def test_unknown_vector_store_is_refused():
    with pytest.raises(ValueError):
        EarningsKnowledgeBaseStack(cdk.App(), "KB", vector_store="pinecone")
