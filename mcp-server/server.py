#!/usr/bin/env python3
"""
MCP Server for OpenSearch Serverless -> Amazon S3 Vectors migration.

Exposes tools to discover OpenSearch Serverless (AOSS) collections/indexes,
migrate their vector data into Amazon S3 Vectors, and run semantic queries
against a migrated S3 Vectors index (optionally with a Bedrock-generated answer).

Authentication uses standard boto3 credential resolution (AWS CLI profiles,
environment variables, or instance/role credentials). No credentials are
stored or transmitted by this server.

Defaults (profile, region, model IDs) can be overridden with environment
variables, which is how the mcp.json "env" block configures the server:
    AWS_PROFILE, AWS_REGION, CHAT_MODEL_ID, EMBED_MODEL_ID
"""

import asyncio
import json
import os
import time
from enum import Enum
from typing import Optional, List, Dict, Any

import boto3
import botocore
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("s3vectors_migration_mcp")

# ─── Constants (overridable via environment) ───
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_PROFILE = os.environ.get("AWS_PROFILE", "default")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
# Cross-Region inference profile (the "us." prefix) keeps this on an active,
# non-legacy model. Matches the default used by the Streamlit app.
CHAT_MODEL_ID = os.environ.get("CHAT_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
MAX_PUT_BATCH = 500  # S3 Vectors PutVectors hard limit


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


# ─── Shared AWS helpers (synchronous; wrapped with asyncio.to_thread in tools) ───

def _session(profile: str, region: str) -> boto3.Session:
    return boto3.Session(profile_name=profile, region_name=region)


def _aoss_client(profile: str, region: str, host: str) -> OpenSearch:
    creds = _session(profile, region).get_credentials().get_frozen_credentials()
    auth = AWSV4SignerAuth(creds, region, "aoss")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )


def _collection_host(collection_id: str, region: str) -> str:
    return f"{collection_id}.{region}.aoss.amazonaws.com"


def _handle_aws_error(e: Exception) -> str:
    """Consistent, actionable error formatting."""
    msg = str(e)
    if isinstance(e, botocore.exceptions.NoCredentialsError):
        return ("Error: No AWS credentials found. Configure a profile with "
                "'aws configure --profile <name>' and pass it as 'profile'.")
    if isinstance(e, botocore.exceptions.ProfileNotFound):
        return f"Error: AWS profile not found. {msg}"
    if isinstance(e, botocore.exceptions.ClientError):
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("AccessDeniedException", "AccessDenied", "403") or "403" in msg:
            return ("Error: Access denied. Check the IAM permissions for this principal, and for "
                    "OpenSearch Serverless ensure the principal is in the collection's data access policy.")
        if code in ("ResourceNotFoundException", "NotFoundException", "404"):
            return f"Error: Resource not found. {msg}"
        if "TooManyRequests" in code or "429" in msg:
            return "Error: Rate limit exceeded. Retry with a smaller batch size or wait before retrying."
        return f"Error: AWS request failed ({code or 'ClientError'}): {msg}"
    if "authorization" in msg.lower() or "403" in msg:
        return ("Error: Authorization failed for OpenSearch Serverless. The principal must be listed "
                "in the collection's data access policy with DescribeIndex and ReadDocument permissions.")
    return f"Error: {type(e).__name__}: {msg}"


def _list_collections(profile: str, region: str) -> List[Dict[str, str]]:
    client = _session(profile, region).client("opensearchserverless")
    resp = client.list_collections()
    return [
        {"id": c["id"], "name": c["name"], "host": _collection_host(c["id"], region)}
        for c in resp.get("collectionSummaries", []) if c.get("status") == "ACTIVE"
    ]


def _list_indexes(profile: str, region: str, host: str) -> List[str]:
    client = _aoss_client(profile, region, host)
    return [i["index"] for i in client.cat.indices(format="json") if not i["index"].startswith(".")]


def _describe_index(profile: str, region: str, host: str, index_name: str) -> Dict[str, Any]:
    client = _aoss_client(profile, region, host)
    mapping = client.indices.get_mapping(index=index_name)
    props = mapping.get(index_name, {}).get("mappings", {}).get("properties", {})
    vector_fields, metadata_fields = [], []
    for name, defn in props.items():
        if defn.get("type") == "knn_vector":
            vector_fields.append({"name": name, "dimension": defn.get("dimension", 0)})
        else:
            metadata_fields.append(name)
    return {"vector_fields": vector_fields, "metadata_fields": metadata_fields}


def _list_vector_buckets(profile: str, region: str) -> List[str]:
    client = _session(profile, region).client("s3vectors")
    return [b["vectorBucketName"] for b in client.list_vector_buckets().get("vectorBuckets", [])]


def _list_vector_indexes(profile: str, region: str, bucket: str) -> List[str]:
    client = _session(profile, region).client("s3vectors")
    return [i["indexName"] for i in client.list_indexes(vectorBucketName=bucket).get("indexes", [])]


def _flatten_metadata(source: dict, vector_field: str, metadata_fields: Optional[List[str]]) -> dict:
    """S3 Vectors metadata supports string/number/boolean only; serialize the rest."""
    out = {}
    for key, value in source.items():
        if key == vector_field:
            continue
        if metadata_fields and key not in metadata_fields:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif value is not None:
            out[key] = json.dumps(value, default=str)
    return out


def _run_migration(profile: str, region: str, host: str, index_name: str, vector_field: str,
                   dimension: int, bucket: str, vector_index: str, distance_metric: str,
                   non_filterable_keys: Optional[List[str]], metadata_fields: Optional[List[str]],
                   batch_size: int, page_size: int) -> Dict[str, Any]:
    """Create the destination bucket/index (verifying ownership) then stream vectors over."""
    session = _session(profile, region)
    s3v = session.client("s3vectors")
    log: List[str] = []

    # Refuse to overwrite an existing index (avoids duplicate data)
    try:
        for idx in s3v.list_indexes(vectorBucketName=bucket).get("indexes", []):
            if idx["indexName"] == vector_index:
                raise ValueError(
                    f"Vector index '{vector_index}' already exists in bucket '{bucket}'. "
                    "Delete it or choose a different name to avoid duplicate data."
                )
    except botocore.exceptions.ClientError:
        pass  # bucket may not exist yet

    # Create bucket, verifying ownership if it already exists (bucket-squatting guard)
    try:
        s3v.create_vector_bucket(vectorBucketName=bucket)
        log.append(f"Created vector bucket: {bucket}")
    except botocore.exceptions.ClientError as e:
        if any(k in str(e).lower() for k in ["already", "exists", "conflict"]):
            s3v.get_vector_bucket(vectorBucketName=bucket)  # raises if not owned/accessible
            log.append(f"Reusing existing vector bucket (ownership verified): {bucket}")
        else:
            raise

    # Create index
    params: Dict[str, Any] = dict(
        vectorBucketName=bucket, indexName=vector_index,
        dimension=dimension, distanceMetric=distance_metric, dataType="float32",
    )
    if non_filterable_keys:
        params["metadataConfiguration"] = {"nonFilterableMetadataKeys": non_filterable_keys}
    s3v.create_index(**params)
    log.append(f"Created vector index: {vector_index} (dim={dimension}, metric={distance_metric})")

    # Stream from OpenSearch using search_after (Scroll API is unsupported on AOSS)
    os_client = _aoss_client(profile, region, host)
    try:
        total_docs = os_client.count(index=index_name).get("count", 0)
    except Exception:
        total_docs = 0

    exported = imported = 0
    buffer: List[dict] = []
    search_after = None

    def flush(vectors: List[dict]) -> int:
        for attempt in range(3):
            try:
                s3v.put_vectors(vectorBucketName=bucket, indexName=vector_index, vectors=vectors)
                return len(vectors)
            except botocore.exceptions.ClientError as e:
                if ("TooManyRequests" in str(e) or "429" in str(e)) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
        return 0

    while True:
        body = {"query": {"match_all": {}}, "size": page_size, "sort": [{"_id": "asc"}]}
        if search_after:
            body["search_after"] = search_after
        resp = os_client.search(index=index_name, body=body)
        hits = resp["hits"]["hits"]
        if not hits:
            break
        for doc in hits:
            vec = doc["_source"].get(vector_field)
            if vec is None:
                continue
            buffer.append({
                "key": doc["_id"],
                "data": {"float32": vec},
                "metadata": _flatten_metadata(doc["_source"], vector_field, metadata_fields),
            })
            exported += 1
            if len(buffer) >= batch_size:
                imported += flush(buffer)
                buffer = []
        search_after = hits[-1]["sort"]

    if buffer:
        imported += flush(buffer)

    log.append(f"Exported {exported} vectors from OpenSearch")
    log.append(f"Imported {imported} vectors into S3 Vectors")
    return {
        "bucket": bucket, "index": vector_index, "dimension": dimension,
        "distance_metric": distance_metric, "source_documents": total_docs,
        "exported": exported, "imported": imported, "log": log,
    }


def _query_vectors(profile: str, region: str, bucket: str, index_name: str,
                   query_text: str, top_k: int, generate_answer: bool) -> Dict[str, Any]:
    session = _session(profile, region)
    bedrock = session.client("bedrock-runtime")
    s3v = session.client("s3vectors")

    resp = bedrock.invoke_model(modelId=EMBED_MODEL_ID, body=json.dumps({"inputText": query_text}))
    embedding = json.loads(resp["body"].read())["embedding"]

    results = s3v.query_vectors(
        vectorBucketName=bucket, indexName=index_name,
        queryVector={"float32": embedding}, topK=top_k,
        returnMetadata=True, returnDistance=True,
    )

    matches, chunks = [], []
    for i, vec in enumerate(results.get("vectors", []), 1):
        meta = vec.get("metadata", {})
        text = meta.get("AMAZON_BEDROCK_TEXT_CHUNK", "") or meta.get("AMAZON_BEDROCK_TEXT", "")
        source = meta.get("x-amz-bedrock-kb-source-uri", "unknown")
        distance = vec.get("distance")
        matches.append({"key": vec.get("key"), "source": source, "distance": distance,
                        "text_preview": text[:200]})
        if text:
            chunks.append(f"[Source {i}: {source}]\n{text}")

    answer = None
    if generate_answer and chunks:
        context = "\n\n---\n\n".join(chunks)
        llm = bedrock.invoke_model(
            modelId=CHAT_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31", "max_tokens": 2048,
                "messages": [{"role": "user", "content":
                    "Answer the question using only the context below. "
                    "Do not include source citations in the answer. "
                    "If the context is insufficient, say so.\n\n"
                    f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"}],
            }),
        )
        answer = json.loads(llm["body"].read())["content"][0]["text"]

    return {"query": query_text, "match_count": len(matches), "matches": matches, "answer": answer}


# ─── Pydantic input models ───

class _AwsBase(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    profile: str = Field(default=DEFAULT_PROFILE, description="AWS CLI profile name to authenticate with (e.g. 'default').")
    region: str = Field(default=DEFAULT_REGION, description="AWS Region of the resources (e.g. 'us-east-1').")


class ListCollectionsInput(_AwsBase):
    pass


class ListIndexesInput(_AwsBase):
    collection_id: str = Field(..., description="OpenSearch Serverless collection ID (e.g. 'abc123xyz'), not the full endpoint.", min_length=1)


class DescribeIndexInput(_AwsBase):
    collection_id: str = Field(..., description="OpenSearch Serverless collection ID.", min_length=1)
    index_name: str = Field(..., description="Name of the index to inspect.", min_length=1)


class ListVectorBucketsInput(_AwsBase):
    pass


class ListVectorIndexesInput(_AwsBase):
    bucket: str = Field(..., description="S3 vector bucket name.", min_length=3, max_length=63)


class MigrateInput(_AwsBase):
    collection_id: str = Field(..., description="Source OpenSearch Serverless collection ID.", min_length=1)
    index_name: str = Field(..., description="Source OpenSearch index containing the vectors.", min_length=1)
    vector_field: str = Field(..., description="Name of the knn_vector field in the source index (see describe_opensearch_index).", min_length=1)
    dest_bucket: str = Field(..., description="Destination S3 vector bucket name (3-63 chars, lowercase, numbers, hyphens). Prefix with your AWS account ID to avoid name collisions. Created if it doesn't exist; ownership is verified if it does.", min_length=3, max_length=63)
    dest_index: str = Field(..., description="Destination S3 vector index name (3-63 chars).", min_length=3, max_length=63)
    dimension: Optional[int] = Field(default=None, description="Vector dimension (1-4096). If omitted, auto-detected from the source index mapping.", ge=1, le=4096)
    distance_metric: str = Field(default="cosine", description="Similarity metric: 'cosine' or 'euclidean'.", pattern=r"^(cosine|euclidean)$")
    metadata_fields: Optional[List[str]] = Field(default=None, description="Metadata field names to carry over. If omitted, all non-vector fields are migrated.")
    non_filterable_keys: Optional[List[str]] = Field(default=None, description="Metadata keys stored as non-filterable (up to 40 KB total vs 2 KB filterable limit). Use for large text chunks. Max 10 keys.", max_length=10)
    batch_size: int = Field(default=100, description="Vectors per PutVectors call (max 500).", ge=1, le=MAX_PUT_BATCH)
    page_size: int = Field(default=500, description="Documents per OpenSearch search_after page (max 10000).", ge=10, le=10000)


class QueryInput(_AwsBase):
    bucket: str = Field(..., description="S3 vector bucket to query.", min_length=3, max_length=63)
    index_name: str = Field(..., description="S3 vector index to query.", min_length=3, max_length=63)
    query_text: str = Field(..., description="Natural-language query. Embedded with Titan Text Embeddings V2 before search.", min_length=1)
    top_k: int = Field(default=5, description="Number of nearest matches to return (1-100).", ge=1, le=100)
    generate_answer: bool = Field(default=True, description="If true, generate a natural-language answer with Claude on Bedrock from the retrieved context. If false, return only the raw matches.")


# ─── Tools ───

@mcp.tool(
    name="list_opensearch_collections",
    annotations={"title": "List OpenSearch Serverless Collections", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def list_opensearch_collections(params: ListCollectionsInput) -> str:
    """List active Amazon OpenSearch Serverless (AOSS) collections in the account/Region.

    Use this first to discover source collections available for migration.

    Args:
        params (ListCollectionsInput): profile, region.

    Returns:
        str: JSON with schema:
        {"count": int, "collections": [{"id": str, "name": str, "host": str}]}
        or "Error: <message>".
    """
    try:
        cols = await asyncio.to_thread(_list_collections, params.profile, params.region)
        return json.dumps({"count": len(cols), "collections": cols}, indent=2)
    except Exception as e:
        return _handle_aws_error(e)


@mcp.tool(
    name="list_opensearch_indexes",
    annotations={"title": "List OpenSearch Indexes", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def list_opensearch_indexes(params: ListIndexesInput) -> str:
    """List user indexes in an OpenSearch Serverless collection (system indexes starting with '.' are omitted).

    Args:
        params (ListIndexesInput): profile, region, collection_id.

    Returns:
        str: JSON {"collection_id": str, "count": int, "indexes": [str]} or "Error: <message>".
        A 403/authorization error means the principal is not in the collection's data access policy.
    """
    try:
        host = _collection_host(params.collection_id, params.region)
        idx = await asyncio.to_thread(_list_indexes, params.profile, params.region, host)
        return json.dumps({"collection_id": params.collection_id, "count": len(idx), "indexes": idx}, indent=2)
    except Exception as e:
        return _handle_aws_error(e)


@mcp.tool(
    name="describe_opensearch_index",
    annotations={"title": "Describe OpenSearch Index Fields", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def describe_opensearch_index(params: DescribeIndexInput) -> str:
    """Inspect an OpenSearch index mapping to find its vector field(s) and metadata fields.

    Use this before migrating to determine the correct 'vector_field' and its dimension.

    Args:
        params (DescribeIndexInput): profile, region, collection_id, index_name.

    Returns:
        str: JSON with schema:
        {
          "index": str,
          "vector_fields": [{"name": str, "dimension": int}],
          "metadata_fields": [str]
        }
        or "Error: <message>".
    """
    try:
        host = _collection_host(params.collection_id, params.region)
        info = await asyncio.to_thread(_describe_index, params.profile, params.region, host, params.index_name)
        return json.dumps({"index": params.index_name, **info}, indent=2)
    except Exception as e:
        return _handle_aws_error(e)


@mcp.tool(
    name="list_s3_vector_buckets",
    annotations={"title": "List S3 Vector Buckets", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def list_s3_vector_buckets(params: ListVectorBucketsInput) -> str:
    """List Amazon S3 vector buckets in the account/Region.

    Args:
        params (ListVectorBucketsInput): profile, region.

    Returns:
        str: JSON {"count": int, "buckets": [str]} or "Error: <message>".
    """
    try:
        buckets = await asyncio.to_thread(_list_vector_buckets, params.profile, params.region)
        return json.dumps({"count": len(buckets), "buckets": buckets}, indent=2)
    except Exception as e:
        return _handle_aws_error(e)


@mcp.tool(
    name="list_s3_vector_indexes",
    annotations={"title": "List S3 Vector Indexes", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def list_s3_vector_indexes(params: ListVectorIndexesInput) -> str:
    """List vector indexes inside an S3 vector bucket.

    Args:
        params (ListVectorIndexesInput): profile, region, bucket.

    Returns:
        str: JSON {"bucket": str, "count": int, "indexes": [str]} or "Error: <message>".
    """
    try:
        idx = await asyncio.to_thread(_list_vector_indexes, params.profile, params.region, params.bucket)
        return json.dumps({"bucket": params.bucket, "count": len(idx), "indexes": idx}, indent=2)
    except Exception as e:
        return _handle_aws_error(e)


@mcp.tool(
    name="migrate_opensearch_to_s3_vectors",
    annotations={"title": "Migrate OpenSearch Vectors to S3 Vectors", "readOnlyHint": False,
                 "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def migrate_opensearch_to_s3_vectors(params: MigrateInput, ctx: Context) -> str:
    """Migrate all vectors from an OpenSearch Serverless index into a new S3 Vectors index.

    Creates the destination bucket (verifying ownership if it already exists) and index, then
    streams every document from the source index using search_after pagination and writes them
    to S3 Vectors in batches. Fails fast if the destination index already exists to avoid
    duplicate data.

    If 'non_filterable_keys' is omitted, the tool auto-classifies any source metadata field whose
    name contains "TEXT" (e.g. AMAZON_BEDROCK_TEXT, AMAZON_BEDROCK_TEXT_CHUNK) as non-filterable.
    This prevents PutVectors 400 errors, since filterable metadata is capped at 2 KB per vector
    while non-filterable metadata allows up to 40 KB. Pass an explicit list to override, or an
    empty list to force everything filterable.

    Args:
        params (MigrateInput): profile, region, collection_id, index_name, vector_field,
            dest_bucket, dest_index, dimension (optional/auto-detected), distance_metric,
            metadata_fields (optional), non_filterable_keys (optional), batch_size, page_size.

    Returns:
        str: JSON with schema:
        {
          "bucket": str, "index": str, "dimension": int, "distance_metric": str,
          "source_documents": int, "exported": int, "imported": int, "log": [str]
        }
        or "Error: <message>".
    """
    try:
        host = _collection_host(params.collection_id, params.region)

        dimension = params.dimension
        non_filterable = params.non_filterable_keys

        # Fetch the source mapping if we need to auto-detect the dimension and/or
        # the non-filterable metadata keys.
        if dimension is None or non_filterable is None:
            await ctx.info("Reading source index mapping for auto-detection")
            info = await asyncio.to_thread(_describe_index, params.profile, params.region, host, params.index_name)

            if dimension is None:
                match = next((vf for vf in info["vector_fields"] if vf["name"] == params.vector_field), None)
                if match is None:
                    available = [vf["name"] for vf in info["vector_fields"]]
                    return (f"Error: vector_field '{params.vector_field}' not found in index "
                            f"'{params.index_name}'. Available knn_vector fields: {available}")
                dimension = match["dimension"]

            if non_filterable is None:
                # Large text fields (chunks) routinely exceed the 2 KB filterable metadata
                # limit and would fail PutVectors with a 400. Auto-classify any metadata
                # field whose name contains "TEXT" as non-filterable (allows up to 40 KB).
                # Only fields present in the source are used, respecting an explicit
                # metadata_fields filter, capped at the 10-key limit.
                candidates = params.metadata_fields or info["metadata_fields"]
                non_filterable = [f for f in candidates if "TEXT" in f.upper()][:10]
                if non_filterable:
                    await ctx.info(f"Auto-selected non-filterable keys: {non_filterable}")

        await ctx.info(f"Starting migration into s3://{params.dest_bucket}/{params.dest_index}")
        result = await asyncio.to_thread(
            _run_migration, params.profile, params.region, host, params.index_name,
            params.vector_field, dimension, params.dest_bucket, params.dest_index,
            params.distance_metric, non_filterable, params.metadata_fields,
            params.batch_size, params.page_size,
        )
        result["non_filterable_keys"] = non_filterable
        return json.dumps(result, indent=2)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return _handle_aws_error(e)


@mcp.tool(
    name="query_s3_vector_store",
    annotations={"title": "Query S3 Vector Store (RAG)", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def query_s3_vector_store(params: QueryInput) -> str:
    """Semantically search an S3 Vectors index and optionally generate an answer with Bedrock.

    Embeds the query with Titan Text Embeddings V2, retrieves the nearest vectors, and (when
    generate_answer is true) passes the retrieved text to Claude on Bedrock to produce an answer.

    Args:
        params (QueryInput): profile, region, bucket, index_name, query_text, top_k, generate_answer.

    Returns:
        str: JSON with schema:
        {
          "query": str,
          "match_count": int,
          "matches": [{"key": str, "source": str, "distance": float, "text_preview": str}],
          "answer": str | null
        }
        or "Error: <message>".
    """
    try:
        result = await asyncio.to_thread(
            _query_vectors, params.profile, params.region, params.bucket,
            params.index_name, params.query_text, params.top_k, params.generate_answer,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return _handle_aws_error(e)


if __name__ == "__main__":
    mcp.run()
