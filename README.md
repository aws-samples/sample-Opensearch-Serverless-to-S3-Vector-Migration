# OpenSearch Serverless → S3 Vectors Migration Tool

Migrate vector data from Amazon OpenSearch Serverless (AOSS) to Amazon S3 Vectors, with a built-in
chatbot to test the migrated data. The project ships two interchangeable front ends over the same
migration and query logic:

- **Streamlit web app** (`app.py`) — a point-and-click UI.
- **MCP server** (`mcp-server/`) — the same capabilities exposed as agent-callable tools over the
  Model Context Protocol, for use from Kiro, Claude Desktop, or any MCP client.

![OpenSearch to S3 Vectors Migration Tool](screenshot.png)

## Repository layout

```
.
├── app.py                 # Streamlit web app
├── requirements.txt       # Streamlit app dependencies
├── mcp-server/            # MCP server (agent-callable tools)
│   ├── server.py
│   ├── requirements.txt
│   ├── mcp.json.example
│   └── README.md          # MCP-specific setup and configuration
├── screenshot.png
└── README.md
```

## Features

- Auto-discovers AOSS collections, indexes, and field mappings from your AWS CLI profile
- Auto-detects vector dimensions and metadata fields from the index mapping
- Creates S3 vector buckets and indexes with configurable distance metrics and non-filterable metadata keys
- Streams vectors from OpenSearch to S3 Vectors using `search_after` pagination (Scroll API is not supported on AOSS)
- Batched `PutVectors` calls with throttle retry handling
- Built-in chatbot to query the migrated S3 Vectors index using Titan Embeddings V2 + Claude on Bedrock
- Available as both a Streamlit app and an MCP server

## Prerequisites

1. **Python 3.10+**
2. **AWS CLI** installed and configured with at least one named profile (`aws configure`)
3. **IAM permissions** for OpenSearch Serverless (`aoss:ListCollections`, `aoss:BatchGetCollection`, `aoss:APIAccessAll`), S3 Vectors (`s3vectors:CreateVectorBucket`, `CreateIndex`, `PutVectors`, `QueryVectors`, and list/get actions), and `bedrock:InvokeModel` — scoped to your specific collection and vector bucket rather than `*`. See the in-app **IAM Permissions** panel for a ready-to-use least-privilege policy.
4. **AOSS Data Access Policy** granting your IAM principal `aoss:DescribeIndex` and `aoss:ReadDocument` on the source collection

## Option A — Streamlit web app

### Setup

Using a virtual environment keeps dependencies isolated. Commands differ by platform.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If PowerShell blocks the activation script, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` for the current session, then
> re-run the activate command.

**Windows (Command Prompt)**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Run

The `streamlit` command is the same on every platform:

```bash
streamlit run app.py
```

The app provides a full UI with:
- Cascading dropdowns for collection → index → vector field selection
- Metadata field picker with non-filterable key configuration
- S3 Vectors destination settings (bucket, index, distance metric, batch size)
- Migration progress bar with success/failure reporting
- Chatbot panel to semantically search the migrated vector store

## Option B — MCP server

The MCP server exposes the same discovery, migration, and query logic as agent-callable tools. See
**[`mcp-server/README.md`](mcp-server/README.md)** for full setup, platform-specific configuration,
and the tool reference.

Quick start:

**macOS / Linux**

```bash
cd mcp-server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
cd mcp-server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then copy `mcp-server/mcp.json.example` into your MCP client config, set the absolute path to
`server.py`, and adjust the `env` block (profile, region, model IDs). macOS/Linux use a
forward-slash path and `python3`; Windows uses an escaped `C:\\...` path and `python`. The
MCP server README shows both formats in full.

## Key Technical Details

- **AOSS does not support the Scroll API.** The tool uses `search_after` with `_id` sort for deep pagination.
- **AOSS uses IAM SigV4 auth**, not username/password. The app uses `AWSV4SignerAuth` from `opensearch-py`.
- **S3 Vectors `PutVectors`** supports max 500 vectors per batch, 1,000 requests/sec, 2,500 vectors/sec per index.
- **Metadata types:** S3 Vectors supports string, number, and boolean only. Nested objects are serialized to JSON strings.
- **Filterable metadata** is limited to 2 KB per vector. Large text fields (like text chunks) should be configured as non-filterable keys (up to 40 KB total).
- **Vector dimensions** must be 1–4,096 (float32 only). The app auto-detects this from the OpenSearch index mapping.
- The chatbot LLM is configurable and defaults to the Claude Sonnet 4.5 cross-Region inference profile (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`). Current Claude models must be invoked via an inference profile ID (prefixed `us.`, `eu.`, etc.), not the raw foundation-model ID, and must be enabled under Bedrock → Model access. In the Streamlit app it is set in the sidebar; in the MCP server it is set via the `CHAT_MODEL_ID` env var.

## References

- [S3 Vectors User Guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)
- [S3 Vectors Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-best-practices.html)
- [S3 Vectors Limitations](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-limitations.html)
- [AOSS Supported Operations](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-genref.html)
- [AOSS Data Access Policies](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-data-access.html)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
