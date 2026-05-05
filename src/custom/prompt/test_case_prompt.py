from custom.utils.test_utils import offset_date

SINGLE_SERVER_MULTI_TOOL_PROMPT = '''## Task  
Generate a **Tool Use Question** based on the provided MCP Server and its tool descriptions.

## Objective  
Analyze the given MCP Server and its available tools, then create a realistic user question that naturally requires **one or more tool calls** (from this same server) to resolve.

## Guidelines  

### Question Realism  
- The question should reflect a practical, real-world scenario that involves a **multi-step workflow** or **combined data/action requirements** (e.g., “Check if a deployment is healthy, and if so, scale it up”).  
- Phrase it naturally—as if asked by a real user—without mentioning tool names, APIs, or internal system details.  
- Include sufficient contextual information (e.g., resource IDs, time ranges, statuses, names) to **unambiguously determine all needed tool calls and their parameters**.

### Tool Selection  
- You may select **one or more tools** from the same MCP Server.  
- The sequence and combination of tools should reflect a **logical workflow** (e.g., fetch → validate → act, or parallel data gathering).  

### Question Complexity  
- The question must be complex enough to **require multiple distinct tool invocations** (or at least allow for a multi-tool resolution path), yet still be **fully resolvable using only the provided tools**.  
- Avoid questions that can be answered with general knowledge or a single data point.  
- **Do not include any tool or parameter names** in the user’s question—keep it task-focused and user-centric.

### Question Content
- The question can be writen in English or in the same language as the tool description (such as Chinese).
- Regarding real-time issues, the time is set around {set_time}.
- The questions should be as diverse as possible, covering the functions of the MCP server from multiple perspectives

### Output Format  
Your response must include:  
1. **Server Analysis**: A concise summary of the MCP Server’s purpose and the key capabilities of its tools.  
2. **Target Tools**: A list of the **names of one or more tools** that would be required to answer the question.
3. **Question**: A natural, realistic user question that justifies the use of these tools.

## MCP Server Description  
`{MCP_SERVER_NAME}`: `{MCP_SERVER_DESCRIPTION}`  

Available Tools:  
`{TOOL_LIST}`  

## Environment
These files are stored in the local file system and serve as relevant resources for tool utilization. If a tool requires access to these files, queries may be formulated incorporating them as contextual inputs. Conversely, if the tool's functionality does not entail these files, these resources may be safely disregarded without affecting operational integrity.
`{FILE_EXISTS}`  

## Output Requirements  
Return your response in the following **XML structure**:

```xml
<response>
  <server_analysis>
    <!-- Briefly summarize the MCP Server's purpose and the main functionalities of its available tools. -->
  </server_analysis>
  <target_tools>
    <tool_name>tool_name_1</tool_name>
    <tool_name>tool_name_2</tool_name>
    <!-- Add more tool_name elements as needed -->
  </target_tools>
  <question>
    <!-- A natural-sounding, realistic user question that requires the above tools to resolve. -->
  </question>
</response>'''

MULTI_SERVER_MULTI_TOOL_PROMPT = '''## Task  
Generate a realistic user question that requires tools from **multiple MCP servers** to resolve.

## Guidelines  
- The question must be concise and should reflect a **practical, real-world workflow** involving **different servers (>=2)**. 
- Include sufficient contextual information to **unambiguously determine all needed tool calls and their parameters**, but do not use real user names. 
- Phrase it naturally—as if asked by a real user—**without mentioning any tool names, server names, or technical internals**.  
- The solution should require **multiple tool invocations across servers**.  
- **Do not use all servers/tools**—instead, select a **coherent subset** that covers distinct capabilities within one realistic scenario.  
- Rely **only on the provided servers and tools**—no external knowledge or assumptions.  
- The question may be written in English or in the same language as the tool descriptions.
- Regarding real-time issues, the time is set around {set_time}.
- Cannot use files that do not exist locally unless you have previously saved them.

## Input  
Available MCP Servers:  
{SERVER_DESCRIPTIONS}  

Relevant files (if applicable):  
{FILE_EXISTS}  

Database Configuration (if applicable):
- Host: 127.0.0.1
- Port: 3305
- User: root
- Password: 123456
- Database Name: mydb

## Output Format  
Return exactly one response in the following XML structure:

```xml
<response>
  <workflow_analysis>
    <!-- Briefly explain how the selected servers interact in this scenario -->
  </workflow_analysis>
  <target_servers>
    <server name="server_name_1">
      <tool>tool_name_a</tool>
      <tool>tool_name_b</tool>
    </server>
    <server name="server_name_2">
      <tool>tool_name_c</tool>
    </server>
    <!-- Add more servers/tools as needed -->
  </target_servers>
  <question>
    <!-- Natural-language user question -->
  </question>
</response>'''

SINGLE_SERVER_MULTI_TOOL_PROMPT = SINGLE_SERVER_MULTI_TOOL_PROMPT.replace('{set_time}', str(offset_date(30)))
MULTI_SERVER_MULTI_TOOL_PROMPT = MULTI_SERVER_MULTI_TOOL_PROMPT.replace('{set_time}', str(offset_date(30)))

SINGLE_SERVER_SINGLE_TOOL_PROMPT='''## Task  
Generate a **Tool Use Question** based on the provided MCP Server and its tool descriptions.

## Objective  
Analyze the given MCP Server and its available tools, then create a realistic user question that naturally requires the use of **exactly one specific tool with concrete input parameters** to solve.

## Guidelines  

### Question Realism  
- The question should reflect a genuine, practical scenario where a user needs to perform a specific action or retrieve specific information via the MCP Server.  
- Phrase it naturally—as if asked by a real user—without mentioning tool names, APIs, or implementation details.  
- Include sufficient context (e.g., user ID, date, resource name, status, etc.) so that the required tool call can be **unambiguously determined**, including parameter values.
- It is best not to complete the generated question with less than 3 tool calls and more than 10 tool calls.

### Tool Selection  
- Select **exactly one tool** that is both necessary and sufficient to answer the question.  
- Based on the user’s stated need and context, infer the **concrete parameter values** that would be passed to the tool.  
- The tool’s signature and description must support the inferred parameters.

### Question Complexity  
- The question must be specific enough to **require tool usage** and to **fully determine the tool’s input parameters**.  
- Avoid vague or open-ended questions (e.g., “What can you do?”) or those lacking key details needed for parameter binding.  
- **Do not include the tool name or parameter names** in the user’s question—keep it user-centric.

### Output Format  
Your response must include:  
1. **Server Analysis**: A brief overview of the MCP Server’s purpose and the key functionalities of its tools.  
2. **Target Tool Call**: A **structured representation** of the single tool invocation, including:
   - The tool name  
   - The exact parameter names and their concrete values (as they would be passed at runtime)  
3. **Question**: A natural, realistic user question that leads directly to this specific tool call.

## MCP Server Description  
`{MCP_SERVER_NAME}`: `{MCP_SERVER_DESCRIPTION}`  

Available Tools:  
`{TOOL_LIST}`  

## Exist Cases
This is an existing test case, please generate a different one:
`{CASE}`  

## Output Requirements  
Return your response in the following **XML structure**:

```xml
<response>
  <server_analysis>
    <!-- Briefly summarize the MCP Server's purpose and the main capabilities of its available tools. -->
  </server_analysis>
  <target_tool_call>
    <tool_name>example_tool_name</tool_name>
    <parameters>
      <parameter name="param1">value1</parameter>
      <parameter name="param2">value2</parameter>
      <!-- Add more parameters as needed -->
    </parameters>
  </target_tool_call>
  <question>
    <!-- A natural-sounding, realistic user question that maps to this exact tool call. -->
  </question>
</response>
```'''

TOOL_ERROR_SOURCE_PROMPT = '''**Role**: You are a QA Automation Expert specializing in Model Context Protocol (MCP) tools evaluation.

**Task**: Analyze the execution of an MCP tool based on the provided context and determine if the test case was successfully resolved. If it failed, you must diagnose the root cause.

**Input Data**:
1. **Tool Description**: {tool_description}
2. **Test Case**: {test_case}
3. **Oracle Tool Call (The arguments passed)**: {tool_call}
4. **Execution Result (The output from the tool)**: {execution_result}

**Evaluation Logic**:
- **Success**: The `tool_call` aligns with the `test_case` intent, and the `execution_result` confirms the desired outcome was achieved according to the `tool_description`.
- **Failure - Parameter Issue**: 
    - The agent provided incorrect parameter types.
    - Required parameters are missing.
    - The logic of the arguments does not match the requirements of the test case.
    - The values provided are out of range or semantically incorrect for the tool's schema.
- **Failure - MCP Issue**: 
    - The tool returned an internal error (500, crash, timeout).
    - The tool's output logic contradicts the `tool_description` even though the input parameters were correct.
    - The tool failed to connect to the underlying service or resource.

**Output Format**:
Return your analysis strictly in the following JSON format:

```json
{
  "is_successful": boolean,
  "failure_category": "None" | "Parameter_Issue" | "MCP_Issue" | "Both",
  "reasoning": "A detailed explanation of why the test passed or failed, comparing the call and result against the description.",
  "comparison": {
    "expected_behavior": "What should have happened based on the test case",
    "actual_behavior": "What actually happened based on the result"
  },
  "suggestions": "Specific advice to fix the issue (if any)."
}'''