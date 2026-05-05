GITHUB_README_COMMAND_EXTRACT = '''You are a precise command-line extractor. From the provided README content, extract **all** executable commands that fall into one of the following categories:

1. **npx commands**:  
   - Must start with `npx` (e.g., `npx my-server`, `npx -y @scope/tool@latest`)

2. **docker commands**:  
   - Must start with `docker` (e.g., `docker run -p 8080:8080 image`, `docker run ...`)

3. **uvx commands**:  
   - Must start with `uvx` (e.g., `uvx tool-name`, `uvx --from pkg tool`)

4. **Other direct execution commands**:  
   - Any other terminal command that directly runs a tool or server, such as:
     - `pipx run package`
     - `python -m module`
     - `go run .`
     - `npm exec tool`
     - `deno run ...`
     - `cargo run -- ...`
   - Must be a full, runnable command

**Exclusions**:
- Build, test, lint, or development scripts (e.g., `npm run build`, `pytest`, `make`)
- Environment variable exports (e.g., `export KEY=value`)
- Windows-specific commands
- Configuration instructions or conceptual descriptions

**Output Format**:
- Output a JSON array of objects.
- Each object must have:
  - `"command"`: the executable name as a string (e.g., `"npx"`, `"docker"`, `"uvx"`, `"python"`, `"pipx"`)
  - `"args"`: a list of all subsequent tokens, exactly as written (preserve flags, paths, options, quoting if any)
- `"full_command"`: the complete command as a single string that can be copied and run directly in shell (formed by joining `command` and `args` with single spaces)

- Do NOT include `"type"` field — category is implicit from `"command"`.
- Do NOT add explanations, comments, or markdown.
- If no commands found, output an empty array: `[]`

**Example Output**:
[
  {
    "command": "npx",
    "args": ["-y", "@playwright/mcp@latest"],
    "full_command": "npx -y @playwright/mcp@latest"
  },
  {
    "command": "docker",
    "args": ["run", "-p", "8080:8080", "ghcr.io/user/mcp-git", "/repo"],
    "full_command": "docker run -p 8080:8080 ghcr.io/modelcontextprotocol/mcp-filesystem /workspace"
  },
  {
    "command": "uvx",
    "args": ["mcp-server-filesystem", "/tmp"],
    "full_command": "uvx mcp-server-git --repository /home/user/project"
  },
  {
    "command": "pipx",
    "args": ["mcp-server-filesystem", "--repository", "/home/user/project"],",
    "full_command": "pipx mcp-server-git --repository /home/user/project""
  }
]
'''