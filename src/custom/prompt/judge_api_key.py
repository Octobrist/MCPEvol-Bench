EXTRACT_CONFIG_PROMPT='''You are an expert MCP (Model Context Protocol) configuration extractor. Analyze the provided README content and generate a **strictly valid JSON output** with two top-level properties:

1. `"config"` – VS Code-compatible MCP server configuration object  
2. `"metadata"` – API key requirement analysis

**Extraction Rules:**

✅ **For `config.mcpServers`:**
- Identify the primary MCP server name from README (e.g., package name, title, or explicit server name)
- Extract launch command pattern (e.g., `npx <package>`, `python -m server`, `docker run...`)
- Split command into:
  - `"command"`: Executable only (e.g., `"npx"`, `"python"`, `"docker"`)
  - `"args"`: Array of arguments (e.g., `["bazi-mcp"]`, `["-m", "my_server"]`)
- If multiple servers exist, include all under `mcpServers` with unique keys
- If command pattern is ambiguous, use the most prominently documented launch method
- Default to empty object `{}` for `mcpServers` if no launch instructions found

✅ **For `metadata.requires_api_key`:**
- `true` ONLY if README explicitly states external API credentials are **required for basic operation** (e.g., "You must set OPENAI_API_KEY to start the server")
- `false` if:
  - Server works locally without external services
  - API keys are only for optional features
  - No credential requirements mentioned
- Be conservative – prefer `false` when uncertain

✅ **For `metadata.api_key_examples`:**
- List ONLY environment variable names explicitly mentioned as API keys (e.g., `["GROQ_API_KEY", "ANTHROPIC_API_KEY"]`)
- Empty array `[]` if none detected

**Output Schema (STRICTLY VALID JSON):**
```json
{
  "config": {
    "mcpServers": {
      "ServerName": {
        "command": "string (npx command)",
        "args": ["string"]
      }
    }
  },
  "metadata": {
    "requires_api_key": boolean,
    "api_key_examples": ["string"]
  }
}
```

**Critical Requirements:**
- NEVER omit top-level keys (`config`, `metadata`)
- `mcpServers` must be an object (not array) with server names as keys
- Output ONLY raw JSON – no explanations, markdown, or prefixes
- Escape special characters properly for valid JSON
- If no server detected: `"mcpServers": {}`
- If no API keys mentioned: `"api_key_examples": []`

**Input README:**
```
{readme_content}
```'''