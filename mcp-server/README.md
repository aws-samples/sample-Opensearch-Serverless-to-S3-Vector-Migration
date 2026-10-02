# S3 Vectors Migration MCP Server

An MCP (Model Context Protocol) server that lets an AI agent discover OpenSearch Serverless
collections, migrate their vector data into Amazon S3 Vectors, and run semantic queries against
the migrated index — all through tool calls.

This is the MCP-server equivalent of the Streamlit app in the parent directory. It exposes the
same underlying migration and query logic as agent-callable tools.

## Tools

| Tool | Type | Description |
|------|------|-------------|
| `list_opensearch_collections` | read-only | List active AOSS collections in the account/Region |
| `list_opensearch_indexes` | read-only | List indexes in a collection |
| `describe_opensearch_index` | read-only | Inspect an index mapping for vector field(s) + dimension and metadata fields |
| `list_s3_vector_buckets` | read-only | List S3 vector buckets |
| `list_s3_vector_indexes` | read-only | List vector indexes in a bucket |
| `migrate_opensearch_to_s3_vectors` | write | Create the destination bucket/index and stream all vectors over |
| `query_s3_vector_store` | read-only | Semantic search + optional Bedrock-generated answer (RAG) |

## Prerequisites

1. **Python 3.10+**
2. **AWS credentials** via an AWS CLI profile, environment variables, or an instance/role. The
   server uses standard boto3 credential resolution and never stores credentials.
3. **IAM permissions** for `aoss` (control plane + data plane on the source collection),
   `s3vectors`, and `bedrock:InvokeModel`. See the parent project's README for a least-privilege
   policy example.
4. **AOSS data access policy** granting your principal `aoss:DescribeIndex` and `aoss:ReadDocument`
   on the source collection.

## Setup

Run these from the `mcp-server` directory. A virtual environment is recommended so the
dependencies stay isolated.

### macOS / Linux

```bash
cd mcp-server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd mcp-server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If PowerShell blocks the activation script, allow it for the current session with
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, then re-run the activate command.

### Windows (Command Prompt)

```bat
cd mcp-server
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Running standalone (optional)

The server uses stdio transport and is normally launched by an MCP client (see below). To test it
in isolation with the MCP Inspector:

### macOS / Linux

```bash
npx @modelcontextprotocol/inspector python3 server.py
```

### Windows (PowerShell)

```powershell
npx @modelcontextprotocol/inspector python server.py
```

## Configuration

Copy `mcp.json.example` into your MCP client config and update the absolute path to `server.py`,
plus your profile/region. The `env` block configures the server defaults
(`AWS_PROFILE`, `AWS_REGION`, `CHAT_MODEL_ID`, `EMBED_MODEL_ID`); any value can still be overridden
per tool call.

For Kiro, the workspace config lives at `.kiro/settings/mcp.json` (or the user-level config at
`~/.kiro/settings/mcp.json`). For Claude Desktop, use its `claude_desktop_config.json`.

### macOS / Linux — path format

Use a forward-slash absolute path:

```json
{
  "mcpServers": {
    "s3vectors-migration": {
      "command": "python3",
      "args": ["/Users/you/path/to/mcp-server/server.py"],
      "env": {
        "AWS_PROFILE": "default",
        "AWS_REGION": "us-east-1",
        "CHAT_MODEL_ID": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "EMBED_MODEL_ID": "amazon.titan-embed-text-v2:0"
      },
      "disabled": false,
      "autoApprove": [
        "list_opensearch_collections",
        "list_opensearch_indexes",
        "describe_opensearch_index",
        "list_s3_vector_buckets",
        "list_s3_vector_indexes",
        "query_s3_vector_store"
      ]
    }
  }
}
```

To use the virtual environment's interpreter explicitly, set `command` to its absolute path —
`/Users/you/path/to/mcp-server/.venv/bin/python` on macOS/Linux.

### Windows — path format

JSON requires backslashes to be escaped (`\\`), so use a double-backslash absolute path and
`python` as the command:

```json
{
  "mcpServers": {
    "s3vectors-migration": {
      "command": "python",
      "args": ["C:\\Users\\you\\path\\to\\mcp-server\\server.py"],
      "env": {
        "AWS_PROFILE": "default",
        "AWS_REGION": "us-east-1",
        "CHAT_MODEL_ID": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "EMBED_MODEL_ID": "amazon.titan-embed-text-v2:0"
      },
      "disabled": false,
      "autoApprove": [
        "list_opensearch_collections",
        "list_opensearch_indexes",
        "describe_opensearch_index",
        "list_s3_vector_buckets",
        "list_s3_vector_indexes",
        "query_s3_vector_store"
      ]
    }
  }
}
```

To use the virtual environment's interpreter explicitly, set `command` to
`C:\\Users\\you\\path\\to\\mcp-server\\.venv\\Scripts\\python.exe`.

Only read-only tools are in `autoApprove`. `migrate_opensearch_to_s3_vectors` is intentionally
left out so it requires explicit approval before writing data.

## Example agent workflow

1. `list_opensearch_collections` → pick a collection ID
2. `list_opensearch_indexes` → pick the source index
3. `describe_opensearch_index` → confirm the vector field name and dimension
4. `migrate_opensearch_to_s3_vectors` → run the migration (dimension auto-detected if omitted)
5. `query_s3_vector_store` → verify the migrated data answers questions correctly

## Notes

- OpenSearch Serverless does not support the Scroll API; the server paginates with `search_after`.
- S3 Vectors `PutVectors` is capped at 500 vectors per batch; migration batches and retries on throttling.
- Large text fields should be passed as `non_filterable_keys` to stay under the 2 KB filterable
  metadata limit (non-filterable allows up to 40 KB).
- If the destination bucket already exists, ownership is verified before any data is written.
- The chat model defaults to the Claude Sonnet 4.5 cross-Region inference profile. Override it with
  the `CHAT_MODEL_ID` env var if a newer inference profile is available, and make sure the model is
  enabled under Bedrock → Model access.
