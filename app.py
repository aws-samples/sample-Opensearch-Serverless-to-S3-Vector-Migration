"""
OpenSearch → S3 Vectors Migration Tool
Streamlit app with migration panel and chatbot for testing.
"""
import json
import time
import boto3
import botocore
import streamlit as st
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

st.set_page_config(
    page_title="OpenSearch → S3 Vectors Migration",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #f8f9fc !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f2f6 100%) !important;
        border-right: 1px solid #e0e3eb;
    }
    section[data-testid="stSidebar"] * { color: #1a1a2e !important; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li { color: #333 !important; }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 60%, #3b82f6 100%);
        padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(37,99,235,0.2);
    }
    .main-header h1 { color: #ffffff !important; margin: 0; font-size: 1.7rem; }
    .main-header p { color: #dbeafe !important; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    .panel-title {
        color: #1e3a5f !important; font-size: 1.35rem; font-weight: 700;
        margin: 0.2rem 0 0.8rem 0; padding-bottom: 0.4rem;
        border-bottom: 3px solid #2563eb;
    }
    .sub-header {
        color: #374151 !important; font-size: 1.0rem; font-weight: 600;
        margin: 1rem 0 0.4rem 0;
    }
    .tooltip-text {
        color: #6b7280 !important; font-size: 0.82rem; font-style: italic; margin-top: -0.5rem;
    }

    .status-box {
        padding: 0.8rem 1rem; border-radius: 8px; margin: 0.5rem 0;
        border-left: 4px solid; font-size: 0.9rem;
    }
    .status-success { background: #ecfdf5; border-color: #10b981; color: #065f46 !important; }
    .status-error { background: #fef2f2; border-color: #ef4444; color: #991b1b !important; }
    .status-info { background: #eff6ff; border-color: #3b82f6; color: #1e40af !important; }
    .status-warn { background: #fffbeb; border-color: #f59e0b; color: #92400e !important; }

    .stSelectbox label, .stTextInput label, .stNumberInput label, .stMultiSelect label {
        color: #1e3a5f !important; font-weight: 500;
    }
    .stMarkdown h3, .stMarkdown h4 { color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ───

@st.cache_data(ttl=120)
def get_aws_profiles():
    try:
        return boto3.Session().available_profiles
    except Exception:
        return ["default"]

# Default Bedrock inference profile for the chatbot LLM.
# Uses a cross-Region inference profile (the "us." prefix) so it stays on an
# active, non-legacy model. Override it in the sidebar if a newer one exists.
DEFAULT_CHAT_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"

def get_regions():
    return ["us-east-1", "us-east-2", "us-west-2", "eu-central-1", "ap-southeast-2"]

def get_boto_session(profile, region):
    return boto3.Session(profile_name=profile, region_name=region)

@st.cache_data(ttl=60, show_spinner=False)
def list_opensearch_collections(profile, region):
    try:
        session = get_boto_session(profile, region)
        client = session.client("opensearchserverless")
        resp = client.list_collections()
        return [
            {"id": c["id"], "name": c["name"], "host": f"{c['id']}.{region}.aoss.amazonaws.com"}
            for c in resp.get("collectionSummaries", []) if c.get("status") == "ACTIVE"
        ]
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=60, show_spinner=False)
def list_opensearch_indexes(profile, region, host):
    try:
        session = get_boto_session(profile, region)
        creds = session.get_credentials().get_frozen_credentials()
        auth = AWSV4SignerAuth(creds, region, "aoss")
        client = OpenSearch(
            hosts=[{"host": host, "port": 443}], http_auth=auth,
            use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection, timeout=30,
        )
        return [i["index"] for i in client.cat.indices(format="json") if not i["index"].startswith(".")]
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=60, show_spinner=False)
def get_index_mapping(profile, region, host, index_name):
    try:
        session = get_boto_session(profile, region)
        creds = session.get_credentials().get_frozen_credentials()
        auth = AWSV4SignerAuth(creds, region, "aoss")
        client = OpenSearch(
            hosts=[{"host": host, "port": 443}], http_auth=auth,
            use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection, timeout=30,
        )
        mapping = client.indices.get_mapping(index=index_name)
        props = mapping.get(index_name, {}).get("mappings", {}).get("properties", {})
        vf, mf = [], []
        for name, defn in props.items():
            if defn.get("type") == "knn_vector":
                vf.append({"name": name, "dimension": defn.get("dimension", 0)})
            else:
                mf.append(name)
        return {"vector_fields": vf, "metadata_fields": mf}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=60, show_spinner=False)
def list_s3_vector_buckets(profile, region):
    try:
        session = get_boto_session(profile, region)
        client = session.client("s3vectors")
        return [b["vectorBucketName"] for b in client.list_vector_buckets().get("vectorBuckets", [])]
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=60, show_spinner=False)
def list_s3_vector_indexes(profile, region, bucket_name):
    try:
        session = get_boto_session(profile, region)
        client = session.client("s3vectors")
        return [i["indexName"] for i in client.list_indexes(vectorBucketName=bucket_name).get("indexes", [])]
    except Exception as e:
        return {"error": str(e)}

def flatten_metadata(source, vector_field, metadata_fields=None):
    metadata = {}
    for key, value in source.items():
        if key == vector_field:
            continue
        if metadata_fields and key not in metadata_fields:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif value is not None:
            metadata[key] = json.dumps(value, default=str)
    return metadata


def run_migration(profile, region, host, index_name, vector_field, dimension,
                  bucket_name, vector_index_name, distance_metric,
                  non_filterable_keys, metadata_fields, batch_size, page_size):
    session = get_boto_session(profile, region)
    s3v = session.client("s3vectors")
    progress = st.progress(0, text="Starting migration...")
    log = []

    try:
        existing = s3v.list_indexes(vectorBucketName=bucket_name)
        for idx in existing.get("indexes", []):
            if idx["indexName"] == vector_index_name:
                return False, (
                    f"Vector index **'{vector_index_name}'** already exists in bucket **'{bucket_name}'**.\n\n"
                    "Delete it first or choose a different name to avoid duplicate data."
                )
    except Exception:
        pass

    progress.progress(5, text="Creating S3 vector bucket...")
    try:
        s3v.create_vector_bucket(vectorBucketName=bucket_name)
        log.append("✅ Created vector bucket: " + bucket_name)
    except botocore.exceptions.ClientError as e:
        if any(k in str(e).lower() for k in ["already", "exists", "bucketalreadyexists"]):
            log.append("ℹ️ Vector bucket already exists: " + bucket_name)
        else:
            return False, f"Failed to create vector bucket:\n`{e}`"

    progress.progress(10, text="Creating vector index...")
    try:
        params = dict(vectorBucketName=bucket_name, indexName=vector_index_name,
                      dimension=dimension, distanceMetric=distance_metric, dataType="float32")
        if non_filterable_keys:
            params["metadataConfiguration"] = {"nonFilterableMetadataKeys": non_filterable_keys}
        s3v.create_index(**params)
        log.append(f"✅ Created vector index: {vector_index_name} (dim={dimension}, metric={distance_metric})")
    except botocore.exceptions.ClientError as e:
        return False, f"Failed to create vector index:\n`{e}`"

    progress.progress(15, text="Connecting to OpenSearch...")
    try:
        creds = session.get_credentials().get_frozen_credentials()
        auth = AWSV4SignerAuth(creds, region, "aoss")
        os_client = OpenSearch(
            hosts=[{"host": host, "port": 443}], http_auth=auth,
            use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection, timeout=60,
        )
    except Exception as e:
        return False, f"Failed to connect to OpenSearch:\n`{e}`"

    try:
        total_docs = os_client.count(index=index_name).get("count", 0)
        log.append(f"ℹ️ Found {total_docs} documents in OpenSearch")
    except Exception:
        total_docs = 0

    total_exported, total_imported = 0, 0
    batch_buffer, search_after = [], None
    meta_fields = metadata_fields if metadata_fields else None

    while True:
        body = {"query": {"match_all": {}}, "size": page_size, "sort": [{"_id": "asc"}]}
        if search_after:
            body["search_after"] = search_after
        try:
            response = os_client.search(index=index_name, body=body)
        except Exception as e:
            return False, f"OpenSearch search failed:\n`{e}`\n\n" + "\n".join(log)

        hits = response["hits"]["hits"]
        if not hits:
            break
        for doc in hits:
            vec = doc["_source"].get(vector_field)
            if vec is None:
                continue
            metadata = flatten_metadata(doc["_source"], vector_field, meta_fields)
            batch_buffer.append({"key": doc["_id"], "data": {"float32": vec}, "metadata": metadata})
            total_exported += 1
            if len(batch_buffer) >= batch_size:
                try:
                    s3v.put_vectors(vectorBucketName=bucket_name, indexName=vector_index_name, vectors=batch_buffer)
                    total_imported += len(batch_buffer)
                    batch_buffer = []
                except Exception as e:
                    if "TooManyRequests" in str(e) or "429" in str(e):
                        time.sleep(2)
                        s3v.put_vectors(vectorBucketName=bucket_name, indexName=vector_index_name, vectors=batch_buffer)
                        total_imported += len(batch_buffer)
                        batch_buffer = []
                    else:
                        return False, f"PutVectors failed:\n`{e}`\n\n" + "\n".join(log)
                if total_docs > 0:
                    progress.progress(min(15 + int((total_imported / total_docs) * 80), 95),
                                      text=f"Migrating... {total_imported}/{total_docs} vectors")
        search_after = hits[-1]["sort"]

    if batch_buffer:
        try:
            s3v.put_vectors(vectorBucketName=bucket_name, indexName=vector_index_name, vectors=batch_buffer)
            total_imported += len(batch_buffer)
        except Exception as e:
            return False, f"PutVectors failed on final batch:\n`{e}`\n\n" + "\n".join(log)

    progress.progress(100, text="Migration complete!")
    log.append(f"✅ Exported {total_exported} vectors from OpenSearch")
    log.append(f"✅ Imported {total_imported} vectors to S3 Vectors")
    list_s3_vector_buckets.clear()
    list_s3_vector_indexes.clear()
    return True, "\n".join(log)


def query_s3_vectors(profile, region, bucket_name, index_name, query_text,
                     chat_model, embed_model=DEFAULT_EMBED_MODEL, top_k=5):
    session = get_boto_session(profile, region)
    bedrock = session.client("bedrock-runtime")
    s3v = session.client("s3vectors")

    resp = bedrock.invoke_model(modelId=embed_model,
                                body=json.dumps({"inputText": query_text}))
    embedding = json.loads(resp["body"].read())["embedding"]

    results = s3v.query_vectors(
        vectorBucketName=bucket_name, indexName=index_name,
        queryVector={"float32": embedding}, topK=top_k,
        returnMetadata=True, returnDistance=True,
    )

    chunks, sources = [], []
    for i, vec in enumerate(results.get("vectors", []), 1):
        meta = vec.get("metadata", {})
        text = meta.get("AMAZON_BEDROCK_TEXT_CHUNK", "") or meta.get("AMAZON_BEDROCK_TEXT", "")
        source = meta.get("x-amz-bedrock-kb-source-uri", "unknown")
        distance = vec.get("distance", "N/A")
        if text:
            chunks.append(f"[Source {i}: {source}]\n{text}")
            sources.append({"source": source, "distance": distance, "preview": text[:150]})

    if not chunks:
        return "No relevant results found in the vector store.", sources

    context = "\n\n---\n\n".join(chunks)
    try:
        llm_resp = bedrock.invoke_model(
            modelId=chat_model,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31", "max_tokens": 2048,
                "messages": [{"role": "user", "content":
                    f"Based on the following context, answer the user's question. "
                    f"Do NOT include source references or citations in your answer — sources are shown separately. "
                    f"If context is insufficient, say so.\n\n"
                    f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"}],
            }),
        )
        answer = json.loads(llm_resp["body"].read())["content"][0]["text"]
    except Exception as e:
        answer = f"Vector search returned {len(chunks)} results but LLM failed: {e}\n\nContext:\n{context[:1000]}"
    return answer, sources


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📋 Prerequisites")
    st.markdown("Before using this tool, ensure you have:")

    with st.expander("**1. AWS CLI** — installed & configured"):
        st.markdown("""
Install the AWS CLI and configure at least one named profile:

```bash
aws configure --profile default
```

This sets up your `Access Key ID`, `Secret Access Key`, and default region.
The tool uses the profile you select to authenticate all AWS API calls.

[AWS CLI Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
        """)

    with st.expander("**2. IAM Permissions** — required policies"):
        st.markdown("""
Your IAM user/role needs these permissions. Attach as an inline policy.

Replace the placeholders before use:
- `REGION` — e.g. `us-east-1`
- `ACCOUNT_ID` — your 12-digit AWS account ID
- `COLLECTION_ID` — the source AOSS collection ID (from the sidebar dropdown)
- `YOUR-VECTOR-BUCKET` — the destination S3 vector bucket name

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "OpenSearchDiscovery",
      "Effect": "Allow",
      "Action": "aoss:ListCollections",
      "Resource": "*"
    },
    {
      "Sid": "OpenSearchCollectionAccess",
      "Effect": "Allow",
      "Action": [
        "aoss:BatchGetCollection",
        "aoss:APIAccessAll"
      ],
      "Resource": "arn:aws:aoss:REGION:ACCOUNT_ID:collection/COLLECTION_ID"
    },
    {
      "Sid": "S3VectorsDiscoverAndCreate",
      "Effect": "Allow",
      "Action": [
        "s3vectors:ListVectorBuckets",
        "s3vectors:CreateVectorBucket"
      ],
      "Resource": "*"
    },
    {
      "Sid": "S3VectorsBucketOperations",
      "Effect": "Allow",
      "Action": [
        "s3vectors:GetVectorBucket",
        "s3vectors:CreateIndex",
        "s3vectors:GetIndex",
        "s3vectors:ListIndexes",
        "s3vectors:PutVectors",
        "s3vectors:QueryVectors"
      ],
      "Resource": "arn:aws:s3vectors:REGION:ACCOUNT_ID:bucket/YOUR-VECTOR-BUCKET/*"
    },
    {
      "Sid": "BedrockForChatbot",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-text-v2:0",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0",
        "arn:aws:bedrock:*:ACCOUNT_ID:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
      ]
    }
  ]
}
```

**IAM Console** → Users/Roles → Permissions → Add permissions → Create inline policy → JSON tab → paste above.

> **Note:** `aoss:ListCollections` and `s3vectors:ListVectorBuckets`/`CreateVectorBucket` require `Resource: "*"` because they are account-level operations that don't support resource-level scoping. All other actions are scoped to the specific collection and bucket you use.
        """)

    with st.expander("**3. AOSS Data Access Policy** — collection access"):
        st.markdown("""
OpenSearch Serverless has a separate access layer. Your IAM principal must be in a **data access policy** on the collection.

**Steps:**
1. **OpenSearch Serverless Console** → Data access policies
2. Find the policy for your collection (e.g. `bedrock-knowledge-base-*`)
3. Click **Edit**
4. Under **Principals**, add your ARN:
   `arn:aws:iam::<account-id>:user/<username>` or
   `arn:aws:iam::<account-id>:role/<role-name>`
5. Grant permissions:
   - **Collection**: `aoss:DescribeCollectionItems`
   - **Index**: `aoss:DescribeIndex`, `aoss:ReadDocument`
6. **Save** — propagates in ~30 seconds

**Or via CLI:**
```bash
aws opensearchserverless update-access-policy \\
  --type data --name "<policy-name>" \\
  --policy-version "<version>" \\
  --policy file://policy.json
```
        """)

    with st.expander("**4. Python Packages** — required libraries"):
        st.markdown("""
Install the required Python packages:

```bash
pip install streamlit boto3 opensearch-py requests-aws4auth
```

All four are needed for the migration tool and chatbot to function.
        """)

    st.divider()
    st.markdown("### ⚙️ Connection")

    profiles = get_aws_profiles()
    selected_profile = st.selectbox(
        "AWS CLI Profile", profiles,
        index=profiles.index("default") if "default" in profiles else 0,
        help="The AWS CLI profile to use for all operations.",
    )
    selected_region = st.selectbox(
        "AWS Region", get_regions(), index=0,
        help="Region where your OpenSearch collection and S3 Vectors reside.",
    )
    chat_model = st.text_input(
        "Chatbot Model (Bedrock)", value=DEFAULT_CHAT_MODEL,
        help="Bedrock model or cross-Region inference profile ID used to answer chat questions. "
             "Use an inference profile ID (prefixed with 'us.', 'eu.', etc.) for current Claude models. "
             "Must be enabled in your account under Bedrock → Model access.",
    )

    st.divider()
    st.markdown("<p style='color:#9ca3af;font-size:0.78rem;text-align:center;'>"
                "OpenSearch → S3 Vectors Migration Tool v1.0</p>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🔄 OpenSearch → S3 Vectors Migration Tool</h1>
    <p>Migrate vector data from OpenSearch Serverless to Amazon S3 Vectors and test with an integrated chatbot.</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# MAIN LAYOUT: Migration | Divider | Chatbot
# ═══════════════════════════════════════════
migration_col, divider_col, chat_col = st.columns([5, 0.1, 4])

# ───────────────────────────────────────────
# LEFT: Migration Panel
# ───────────────────────────────────────────
with migration_col:
    st.markdown('<p class="panel-title">📤 Migration Configuration</p>', unsafe_allow_html=True)

    st.markdown('<p class="sub-header">Source: OpenSearch Serverless</p>', unsafe_allow_html=True)

    with st.spinner("Loading collections..."):
        collections = list_opensearch_collections(selected_profile, selected_region)

    collections_ok = True
    if isinstance(collections, dict) and "error" in collections:
        st.markdown(
            f'<div class="status-box status-error">Failed to list collections: {collections["error"]}</div>',
            unsafe_allow_html=True)
        collections_ok = False
    elif not collections:
        st.markdown(
            '<div class="status-box status-warn">No active AOSS collections found in this region/profile. '
            'Check your AWS CLI profile and region in the sidebar.</div>',
            unsafe_allow_html=True)
        collections_ok = False

    can_proceed = False
    os_host = None
    selected_index = None
    selected_vector_field = None
    detected_dimension = None
    metadata_fields_list = []
    selected_metadata = []
    non_filterable = []

    if collections_ok:
        collection_names = [f"{c['name']} ({c['id']})" for c in collections]
        selected_collection_idx = st.selectbox(
            "OpenSearch Collection", range(len(collection_names)),
            format_func=lambda i: collection_names[i],
            help="Select the OpenSearch Serverless collection containing your vector data.",
        )
        selected_collection = collections[selected_collection_idx]
        os_host = selected_collection["host"]
        st.markdown(f'<p class="tooltip-text">Endpoint: {os_host}</p>', unsafe_allow_html=True)

        # --- Indexes ---
        with st.spinner("Loading indexes..."):
            indexes = list_opensearch_indexes(selected_profile, selected_region, os_host)

        if isinstance(indexes, dict) and "error" in indexes:
            err = indexes["error"]
            if "403" in err or "authorization" in err.lower():
                st.markdown(
                    '<div class="status-box status-warn">'
                    '⚠️ <b>Access denied</b> for this collection. Your IAM principal is not in the '
                    'AOSS data access policy.<br>See <b>Prerequisite #3</b> in the sidebar for how to fix this. '
                    'Or select a different collection above.'
                    '</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="status-box status-error">Failed to list indexes: {err}</div>',
                    unsafe_allow_html=True)
        elif not indexes:
            st.markdown(
                '<div class="status-box status-warn">No indexes found in this collection.</div>',
                unsafe_allow_html=True)
        else:
            selected_index = st.selectbox(
                "OpenSearch Index", indexes,
                help="The index containing your vector embeddings.",
            )

            with st.spinner("Reading index mapping..."):
                mapping = get_index_mapping(selected_profile, selected_region, os_host, selected_index)

            if isinstance(mapping, dict) and "error" in mapping:
                st.markdown(
                    f'<div class="status-box status-error">Failed to read mapping: {mapping["error"]}</div>',
                    unsafe_allow_html=True)
            else:
                vector_fields = mapping.get("vector_fields", [])
                metadata_fields_list = mapping.get("metadata_fields", [])

                if not vector_fields:
                    st.markdown(
                        '<div class="status-box status-warn">No knn_vector fields found in this index.</div>',
                        unsafe_allow_html=True)
                else:
                    vf_options = [f"{vf['name']} (dim={vf['dimension']})" for vf in vector_fields]
                    sel_vf_idx = st.selectbox(
                        "Vector Field", range(len(vf_options)),
                        format_func=lambda i: vf_options[i],
                        help="The field containing vector embeddings. Dimension is auto-detected from the index mapping.",
                    )
                    selected_vector_field = vector_fields[sel_vf_idx]
                    detected_dimension = selected_vector_field["dimension"]
                    st.markdown(
                        f'<p class="tooltip-text">Auto-detected dimension: {detected_dimension}</p>',
                        unsafe_allow_html=True)

                    selected_metadata = st.multiselect(
                        "Metadata Fields to Migrate", metadata_fields_list,
                        default=metadata_fields_list,
                        help="Select which metadata fields to carry over to S3 Vectors. "
                             "Leave all selected to migrate everything.",
                    )
                    non_filterable = st.multiselect(
                        "Non-Filterable Metadata Keys", selected_metadata,
                        default=[f for f in selected_metadata if "TEXT" in f.upper()],
                        help="Large text fields (like text chunks) should be non-filterable to avoid the 2 KB "
                             "filterable metadata limit. Non-filterable keys get up to 40 KB total. Max 10 keys.",
                    )
                    can_proceed = True

    # --- Destination (always visible) ---
    st.markdown('<p class="sub-header" style="margin-top:1.5rem;">Destination: S3 Vectors</p>',
                unsafe_allow_html=True)

    dest_bucket = st.text_input(
        "S3 Vector Bucket Name", value="my-vector-bucket",
        help="Name for the S3 vector bucket (3-63 chars, lowercase, numbers, hyphens). Created if it doesn't exist.",
    )
    dest_index = st.text_input(
        "S3 Vector Index Name", value="my-vector-index",
        help="Name for the vector index (3-63 chars, lowercase, numbers, hyphens, dots).",
    )
    distance_metric = st.selectbox(
        "Distance Metric", ["cosine", "euclidean"],
        help="Cosine: angular similarity (best for normalized vectors). Euclidean: straight-line distance.",
    )
    col_b, col_p = st.columns(2)
    with col_b:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=500, value=100,
                                     help="Vectors per PutVectors API call (max 500).")
    with col_p:
        page_size = st.number_input("Page Size", min_value=10, max_value=10000, value=500,
                                    help="Documents per OpenSearch search_after page (max 10000).")

    # --- Migrate button ---
    if st.button("🚀 Start Migration", type="primary", use_container_width=True, disabled=not can_proceed):
        if not dest_bucket or not dest_index:
            st.error("Please provide both a bucket name and an index name.")
        elif len(non_filterable) > 10:
            st.error("Maximum 10 non-filterable metadata keys allowed.")
        else:
            success, message = run_migration(
                profile=selected_profile, region=selected_region,
                host=os_host, index_name=selected_index,
                vector_field=selected_vector_field["name"],
                dimension=detected_dimension, bucket_name=dest_bucket,
                vector_index_name=dest_index, distance_metric=distance_metric,
                non_filterable_keys=non_filterable if non_filterable else None,
                metadata_fields=selected_metadata if selected_metadata != metadata_fields_list else None,
                batch_size=batch_size, page_size=page_size,
            )
            if success:
                st.markdown(
                    f'<div class="status-box status-success">{message.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True)
                # Fire balloons exactly once, only on a successful migration.
                # A one-shot flag prevents the animation from replaying on later
                # reruns (e.g. when the chatbot panel reruns on each question).
                st.session_state["_migration_succeeded"] = True
            else:
                st.markdown(
                    f'<div class="status-box status-error">❌ Migration Failed<br><br>'
                    f'{message.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True)

    # Show the celebration animation only on the run where a migration just
    # succeeded, then clear the flag so it never replays on subsequent reruns.
    if st.session_state.pop("_migration_succeeded", False):
        st.balloons()


# ───────────────────────────────────────────
# DIVIDER
# ───────────────────────────────────────────
with divider_col:
    st.markdown('<div style="border-left:2px solid #d1d5db;min-height:600px;margin:0 auto;"></div>',
                unsafe_allow_html=True)


# ───────────────────────────────────────────
# RIGHT: Chatbot Panel (runs as a fragment — only this reruns on chat)
# ───────────────────────────────────────────
@st.fragment
def chatbot_panel():
    st.markdown('<p class="panel-title">💬 Test S3 Vector Store</p>', unsafe_allow_html=True)

    vb_list = list_s3_vector_buckets(selected_profile, selected_region)
    if isinstance(vb_list, dict) and "error" in vb_list:
        st.markdown(
            f'<div class="status-box status-warn">Could not list vector buckets: {vb_list["error"]}</div>',
            unsafe_allow_html=True)
        vb_list = []

    if not vb_list:
        st.markdown(
            '<div class="status-box status-info">'
            'No S3 vector buckets found in this region. Run a migration first to create one.'
            '</div>', unsafe_allow_html=True)
    else:
        chat_bucket = st.selectbox("Vector Bucket", vb_list, key="chat_bucket",
                                   help="Select the S3 vector bucket to query.")

        chat_indexes = list_s3_vector_indexes(selected_profile, selected_region, chat_bucket)
        if isinstance(chat_indexes, dict) and "error" in chat_indexes:
            st.markdown(
                f'<div class="status-box status-warn">Could not list indexes: {chat_indexes["error"]}</div>',
                unsafe_allow_html=True)
            chat_indexes = []

        if not chat_indexes:
            st.markdown(
                '<div class="status-box status-info">No vector indexes in this bucket.</div>',
                unsafe_allow_html=True)
        else:
            chat_index = st.selectbox("Vector Index", chat_indexes, key="chat_index",
                                      help="Select the vector index to search against.")

            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []

            chat_container = st.container(height=420)
            with chat_container:
                if not st.session_state.chat_messages:
                    st.markdown(
                        '<div class="status-box status-info">'
                        '👋 Ask a question to search the vector store. Your query will be embedded with '
                        'Titan Text V2, matched against stored vectors, and answered by Claude on Bedrock.'
                        '</div>', unsafe_allow_html=True)
                for msg in st.session_state.chat_messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        if msg.get("sources"):
                            with st.expander("📎 Sources"):
                                for s in msg["sources"]:
                                    dist = s["distance"]
                                    dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else str(dist)
                                    st.markdown(f"- **{s['source']}** (distance: {dist_str})\n  _{s['preview']}..._")

            user_input = st.chat_input("Ask a question about your data...", key="chat_input")
            if user_input and vb_list and chat_indexes:
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                with st.spinner("Searching vectors and generating answer..."):
                    try:
                        answer, sources = query_s3_vectors(
                            selected_profile, selected_region, chat_bucket, chat_index,
                            user_input, chat_model)
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": answer, "sources": sources})
                    except Exception as e:
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": f"❌ Error: {str(e)}"})
                st.rerun(scope="fragment")

with chat_col:
    chatbot_panel()
