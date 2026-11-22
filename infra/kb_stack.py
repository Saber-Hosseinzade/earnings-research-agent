"""The retrieval side: a Bedrock Knowledge Base over the transcript passages.

Documents (one speaker turn per file, built by ``scripts/build_kb_docs.py``) sit
in a private S3 bucket. The Knowledge Base embeds them with Titan Text
Embeddings V2 and stores the vectors in one of two stores, chosen by the
``vector_store`` CDK context value:

- ``s3vectors`` (default): an S3 vector bucket and index. Costs cents a month
  at this size, so it can stay deployed.
- ``opensearch``: an OpenSearch Serverless collection (VECTORSEARCH, standby
  replicas off). Billed per OCU-hour while it exists: deploy for a demo or an
  evaluation run, then destroy this stack.

Both use the same documents, embedding model and no-chunking data source, so
retrieval results from the two stores can be compared on the same golden set.
Switching stores replaces the Knowledge Base; re-run the sync afterwards.

This stack is independent of the tool and runtime stacks and can be destroyed
on its own.
"""

from __future__ import annotations

from aws_cdk import Aws, CfnOutput, Fn, RemovalPolicy, Stack
from aws_cdk import aws_bedrock as bedrock
from aws_cdk import aws_iam as iam
from aws_cdk import aws_opensearchserverless as aoss
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3vectors as s3vectors
from constructs import Construct

VECTOR_STORES = ("s3vectors", "opensearch")
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
DIMENSIONS = 1024
INDEX_NAME = "earnings-passages"
COLLECTION_NAME = "earnings-kb"  # OpenSearch Serverless: lowercase, 3-32 chars

# Field names the Knowledge Base writes. In S3 Vectors they become metadata on
# each vector, where filterable metadata is capped at 2 KB: the passage text and
# Bedrock's own metadata blob must be non-filterable or ingestion fails.
BEDROCK_TEXT = "AMAZON_BEDROCK_TEXT"
BEDROCK_METADATA = "AMAZON_BEDROCK_METADATA"
OSS_VECTOR_FIELD = "embedding"
OSS_TEXT_FIELD = "AMAZON_BEDROCK_TEXT_CHUNK"


class EarningsKnowledgeBaseStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, vector_store: str = "s3vectors", **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        if vector_store not in VECTOR_STORES:
            raise ValueError(f"vector_store must be one of {VECTOR_STORES}, got {vector_store!r}")

        # Holds transcript text: private, TLS only, emptied on destroy.
        self.docs_bucket = s3.Bucket(
            self,
            "Docs",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        embedding_model_arn = f"arn:{Aws.PARTITION}:bedrock:{Aws.REGION}::foundation-model/{EMBEDDING_MODEL_ID}"
        statements = [
            iam.PolicyStatement(actions=["bedrock:InvokeModel"], resources=[embedding_model_arn]),
            iam.PolicyStatement(actions=["s3:ListBucket"], resources=[self.docs_bucket.bucket_arn]),
            iam.PolicyStatement(actions=["s3:GetObject"], resources=[self.docs_bucket.arn_for_objects("*")]),
        ]

        if vector_store == "s3vectors":
            storage, store_statements, store_deps = self._s3vectors()
        else:
            storage, store_statements, store_deps = None, [], []  # filled after the role exists

        self.kb_role = iam.Role(
            self,
            "KnowledgeBaseRole",
            # Only Bedrock, acting for a knowledge base in this account, may assume it.
            assumed_by=iam.ServicePrincipal(
                "bedrock.amazonaws.com",
                conditions={
                    "StringEquals": {"aws:SourceAccount": Aws.ACCOUNT_ID},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:{Aws.PARTITION}:bedrock:{Aws.REGION}:{Aws.ACCOUNT_ID}:knowledge-base/*"
                    },
                },
            ),
            # Inline, so the permissions exist before the Knowledge Base is created.
            inline_policies={"KnowledgeBase": iam.PolicyDocument(statements=statements + store_statements)},
        )

        if vector_store == "opensearch":
            storage, store_deps = self._opensearch()

        self.knowledge_base = bedrock.CfnKnowledgeBase(
            self,
            "KnowledgeBase",
            name=f"earnings-transcripts-{vector_store}",
            description="Earnings-call passages, one speaker turn per document",
            role_arn=self.kb_role.role_arn,
            knowledge_base_configuration=bedrock.CfnKnowledgeBase.KnowledgeBaseConfigurationProperty(
                type="VECTOR",
                vector_knowledge_base_configuration=bedrock.CfnKnowledgeBase.VectorKnowledgeBaseConfigurationProperty(
                    embedding_model_arn=embedding_model_arn,
                    embedding_model_configuration=bedrock.CfnKnowledgeBase.EmbeddingModelConfigurationProperty(
                        bedrock_embedding_model_configuration=bedrock.CfnKnowledgeBase.BedrockEmbeddingModelConfigurationProperty(
                            dimensions=DIMENSIONS, embedding_data_type="FLOAT32"
                        )
                    ),
                ),
            ),
            storage_configuration=storage,
        )
        for dep in store_deps:
            self.knowledge_base.add_resource_dependency(dep)

        # One file = one passage: the builder already split turns, so no chunking here.
        self.data_source = bedrock.CfnDataSource(
            self,
            "Transcripts",
            name="transcript-passages",
            knowledge_base_id=self.knowledge_base.attr_knowledge_base_id,
            data_deletion_policy="DELETE",
            data_source_configuration=bedrock.CfnDataSource.DataSourceConfigurationProperty(
                type="S3",
                s3_configuration=bedrock.CfnDataSource.S3DataSourceConfigurationProperty(
                    bucket_arn=self.docs_bucket.bucket_arn,
                    bucket_owner_account_id=Aws.ACCOUNT_ID,
                ),
            ),
            vector_ingestion_configuration=bedrock.CfnDataSource.VectorIngestionConfigurationProperty(
                chunking_configuration=bedrock.CfnDataSource.ChunkingConfigurationProperty(chunking_strategy="NONE")
            ),
        )

        CfnOutput(self, "VectorStore", value=vector_store)
        CfnOutput(self, "KnowledgeBaseId", value=self.knowledge_base.attr_knowledge_base_id)
        CfnOutput(self, "DataSourceId", value=self.data_source.attr_data_source_id)
        CfnOutput(self, "DocsBucketName", value=self.docs_bucket.bucket_name)

    def _s3vectors(self):
        bucket = s3vectors.CfnVectorBucket(self, "VectorBucket")
        index = s3vectors.CfnIndex(
            self,
            "VectorIndex",
            vector_bucket_arn=bucket.attr_vector_bucket_arn,
            index_name=INDEX_NAME,
            data_type="float32",
            dimension=DIMENSIONS,
            distance_metric="cosine",
            metadata_configuration=s3vectors.CfnIndex.MetadataConfigurationProperty(
                non_filterable_metadata_keys=[BEDROCK_TEXT, BEDROCK_METADATA]
            ),
        )
        index.add_resource_dependency(bucket)
        statements = [
            iam.PolicyStatement(
                actions=[
                    "s3vectors:PutVectors",
                    "s3vectors:GetVectors",
                    "s3vectors:DeleteVectors",
                    "s3vectors:QueryVectors",
                    "s3vectors:GetIndex",
                ],
                resources=[index.attr_index_arn],
            )
        ]
        storage = bedrock.CfnKnowledgeBase.StorageConfigurationProperty(
            type="S3_VECTORS",
            s3_vectors_configuration=bedrock.CfnKnowledgeBase.S3VectorsConfigurationProperty(
                index_arn=index.attr_index_arn
            ),
        )
        return storage, statements, [index]

    def _opensearch(self):
        """OpenSearch Serverless collection and index. Synth-tested; not yet deployed."""
        encryption = aoss.CfnSecurityPolicy(
            self,
            "CollectionEncryption",
            name=f"{COLLECTION_NAME}-enc",
            type="encryption",
            policy=self.to_json_string({"Rules": [_collection_rule()], "AWSOwnedKey": True}),
        )
        # Public endpoint, still IAM-authenticated: Bedrock reaches it without a VPC.
        network = aoss.CfnSecurityPolicy(
            self,
            "CollectionNetwork",
            name=f"{COLLECTION_NAME}-net",
            type="network",
            policy=self.to_json_string([{"Rules": [_collection_rule()], "AllowFromPublic": True}]),
        )
        collection = aoss.CfnCollection(
            self, "Collection", name=COLLECTION_NAME, type="VECTORSEARCH", standby_replicas="DISABLED"
        )
        collection.add_resource_dependency(encryption)
        collection.add_resource_dependency(network)

        # Data access: the KB role reads and writes; the CloudFormation execution
        # role creates the index below.
        # The synthesizer's ARN holds ${AWS::AccountId}/${AWS::Region} placeholders.
        cfn_exec_role = Fn.sub(self.synthesizer.cloud_formation_execution_role_arn)
        access = aoss.CfnAccessPolicy(
            self,
            "CollectionAccess",
            name=f"{COLLECTION_NAME}-access",
            type="data",
            policy=self.to_json_string(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{COLLECTION_NAME}"],
                                "Permission": ["aoss:DescribeCollectionItems"],
                            },
                            {
                                "ResourceType": "index",
                                "Resource": [f"index/{COLLECTION_NAME}/*"],
                                "Permission": [
                                    "aoss:CreateIndex",
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                    "aoss:UpdateIndex",
                                ],
                            },
                        ],
                        "Principal": [self.kb_role.role_arn, cfn_exec_role],
                    }
                ]
            ),
        )
        self.kb_role.add_to_principal_policy(
            iam.PolicyStatement(actions=["aoss:APIAccessAll"], resources=[collection.attr_arn])
        )

        index = aoss.CfnIndex(
            self,
            "VectorIndex",
            collection_endpoint=collection.attr_collection_endpoint,
            index_name=INDEX_NAME,
            settings=aoss.CfnIndex.IndexSettingsProperty(index=aoss.CfnIndex.IndexProperty(knn=True)),
            mappings=aoss.CfnIndex.MappingsProperty(
                properties={
                    OSS_VECTOR_FIELD: aoss.CfnIndex.PropertyMappingProperty(
                        type="knn_vector",
                        dimension=DIMENSIONS,
                        method=aoss.CfnIndex.MethodProperty(name="hnsw", engine="faiss", space_type="l2"),
                    ),
                    OSS_TEXT_FIELD: aoss.CfnIndex.PropertyMappingProperty(type="text", index=True),
                    BEDROCK_METADATA: aoss.CfnIndex.PropertyMappingProperty(type="text", index=False),
                }
            ),
        )
        index.add_resource_dependency(collection)
        index.add_resource_dependency(access)

        storage = bedrock.CfnKnowledgeBase.StorageConfigurationProperty(
            type="OPENSEARCH_SERVERLESS",
            opensearch_serverless_configuration=bedrock.CfnKnowledgeBase.OpenSearchServerlessConfigurationProperty(
                collection_arn=collection.attr_arn,
                vector_index_name=INDEX_NAME,
                field_mapping=bedrock.CfnKnowledgeBase.OpenSearchServerlessFieldMappingProperty(
                    vector_field=OSS_VECTOR_FIELD, text_field=OSS_TEXT_FIELD, metadata_field=BEDROCK_METADATA
                ),
            ),
        )
        # The role's default policy (aoss:APIAccessAll) must exist before the KB checks the index.
        deps = [index]
        if self.kb_role.node.try_find_child("DefaultPolicy") is not None:
            deps.append(self.kb_role.node.find_child("DefaultPolicy").node.default_child)
        return storage, deps


def _collection_rule() -> dict:
    return {"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"]}

