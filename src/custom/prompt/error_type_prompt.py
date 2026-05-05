ERROR_TYPE_PROMPT='''# Role
You are an expert analyst diagnosing failures in **Model Context Protocol (MCP)** agents caused by server evolution. Your task is to map a specific failure scenario to exactly one of the following **categories**.

# Classification Categories
1.  **Ambiguous Parameter Description**: Vague schema descriptions force the Agent to infer specific data formats or standards, leading to payload mismatches despite valid JSON syntax.
2.  **Parameter Schema Drift**: Parameter additions/removals without clear logic impact cause the Agent to rely on outdated mental models, resulting in syntactically valid but logically ineffective requests.
3.  **Implicit Business Logic Dependency**: Individual parameters pass validation, but their combination violates undocumented cross-parameter constraints (e.g., mutual exclusivity), failing at the business logic layer.
4.  **Tool-Chain Disconnection**: Changes in output formats or data structures of upstream tools prevent the Agent from correctly extracting and passing context to downstream tools, breaking the intended execution flow despite successful individual calls.
5.  **Context State Loss**: The Agent mishandles new stateful tool interactions as stateless operations, omitting critical context propagation between steps.
6.  **Tool Semantic Ambiguity**: Blurred tool boundaries or descriptions cause the Agent to select inappropriate tools or incorrect operational strategies.

The input data provides a comprehensive diagnostic context by combining the specific server modifications (**Evolutions**), the user's intended objective (**Task Goal**) and the agent's failed execution attempt on the evolved server (**Failed Trajectory**) against the verified successful execution attempt from the original pre-evolution server (**Ground Truth Trajectory**), with the synthesized error analysis (**Root Cause Metadata**) provided as a derived insight from this comparison.
# Input Data
- **Evolutions**: `{EVOLUTIONS}`
- **Task Goal**: `{QUESTION}`
- **Failed Trajectory**: `{FAILED_TRAJECTORY}`
- **Ground Truth Trajectory**: `{GT_TRAJECTORY}`
- **Root Cause Metadata**: `{ROOT_CAUSE}`

# Instructions
Select the single category that best describes the root mechanism of the failure. 
Errors analogous to 'unhandled errors in a TaskGroup' stem from failures during tool invocation, specifically attributable to invalid input parameters. 
The objective is to analyze the agent's behavior to pinpoint the specific error category responsible for this failure

# Output (JSON Only)
Return ONLY a valid JSON object. Do not include markdown code blocks (```json) or any explanatory text outside the JSON.

{{
  "error_type": <string>,
  "analysis": "<Briefly analyze why it is classified as this type of error>",
}}'''


ERROR_COMPARISON_ANALYSIS = '''# Role
You are an expert MCP Server Evolution Analyst. Your task is to diagnose why an Agent failed on an **Evolved Server** by comparing its trajectory against a successful **Ground Truth Trajectory** from the Original Server.

# Context: MCP Server Evolution
MCP Servers evolve via API updates, tool renaming, schema changes, or logic refinements.
Only the following MCP Server tools have undergone the following evolution, and there are no other evolutions:
{EVOLUTIONS}

# Input Data
## 1. Task Goal
{QUESTION}

## 2. Predicted Trajectory (Evolved Server) - ❌ FAILED
**Trajectory Steps:**
{FAILED_TRAJECTORY}

**Auto-Evaluated Failure Reason:**
{FAILED_REASON}

## 3. Ground Truth Trajectory (Original Server) - ✅ SUCCESS
**Trajectory Steps:**
{GT_TRAJECTORY}

**Auto-Evaluated Success Reason:**
{SUCCESS_REASON}

# Instructions:
1. Identify the Critical Step in **Predicted Trajectory** where the Agent's path first diverged from the Ground Truth logic.
2. Identify whether the evolution of the Server's Evolution (Function, Parameter, or Description) is related to the cause of the error.

# Output (JSON Only)
Return ONLY a valid JSON object. Do not include markdown code blocks (```json) or any explanatory text outside the JSON.

{{
  "critical_step_index": <integer>,
  "failure_point": "<Brief description of how the agent's action diverged from the ground truth logic>",
  "is_evolution_related": <boolean>,
  "root_cause": {{
    "functional_evolution": "<String describing tool addition/removal/renaming IF applicable, otherwise null>",
    "parameter_change": "<String describing schema/argument shifts IF applicable, otherwise null>",
    "description_update": "<String describing text/doc changes causing confusion IF applicable, otherwise null>",
    "other_reason": "<String explaining the reason if is_evolution_related is false (e.g., 'Agent hallucination', 'Logic error'), otherwise null>"
  }},
  "fix_suggestion": "<Concise, actionable instruction for the Agent to succeed on the evolved server>"
}}'''