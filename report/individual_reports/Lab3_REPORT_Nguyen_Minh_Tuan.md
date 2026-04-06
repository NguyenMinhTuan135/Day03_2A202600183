# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Nguyễn Minh Tuấn
- **Student ID**: 2A202600183
- **Date**: April 6, 2026

---

## I. Technical Contribution (15 Points)

*Describe your specific contribution to the codebase (e.g., implemented a specific tool, fixed the parser, etc.).*

- **Modules Implemented**:
  - `src/agent/agent.py`
  - `src/tools/stock_tools.py`
  - `src/telemetry/logger.py`
  - `src/core/llm_provider.py`

- **Code Highlights**:
  - Implemented the ReAct execution loop in `ReActAgent.run()` to alternate between `Thought`, `Action`, and `Observation`.
  - Added the tool registry and execution flow via `_execute_tool()` so the agent can dispatch stock queries and calculations dynamically.
  - Built the system prompt generator in `get_system_prompt()` with strict format instructions and tool descriptions.
  - Integrated structured telemetry events (`AGENT_START`, `LLM_RESPONSE`, `TOOL_CALL`, `TOOL_RESULT`, `AGENT_END`) using `src/telemetry/logger.py`.

- **Documentation**:
  - The ReAct agent uses the `LLMProvider` interface to generate responses given the current prompt and system instructions.
  - Each cycle extracts the model's `Action` and executes the corresponding tool from `src/tools/stock_tools.py`.
  - The observation result is appended to the prompt, enabling the next iteration to reason from real tool data rather than guessing.

---

## II. Debugging Case Study (10 Points)

*Analyze a specific failure event you encountered during the lab using the logging system.*

- **Problem Description**: When the agent response omitted a valid `Action` or deviated from the exact `Action: tool_name(args)` syntax, the loop terminated early with `status: no_action`, or it reached `max_steps` without producing a final answer.

- **Log Source**: The expected log trace is produced by `src/telemetry/logger.py` and would appear in `logs/YYYY-MM-DD.log` as JSON events such as `LLM_RESPONSE`, `TOOL_CALL`, `TOOL_RESULT`, and `AGENT_END`.

- **Diagnosis**: The failure was caused by the combination of prompt formatting and regex parsing in `ReActAgent.run()`. The model must follow an exact action format, but if the response includes extra punctuation or a missing `Action` block, the parser does not recognize the tool request and the agent treats the output as a terminal answer.

- **Solution**: I fixed the agent by enforcing stricter prompt instructions and adding more robust parsing. A production-grade improvement is to validate the LLM output after each step and, if parsing fails, request a corrected response instead of ending the session.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

*Reflect on the reasoning capability difference.*

1.  **Reasoning**: The `Thought` block forces explicit planning before any tool call. This makes the agent more transparent and helps separate the decision about which tool to use from the actual tool invocation, unlike a direct chatbot answer that may hallucinate or jump to conclusions.

2.  **Reliability**: The Agent can perform worse when the task is very simple or when the LLM violates the required ReAct format. In those cases, a direct chatbot answer is faster and more reliable because it avoids the overhead of tool orchestration and the risk of parser failure.

3.  **Observation**: Environment feedback is critical. After each tool call, the returned `Observation` provides factual data that the next `Thought` uses to decide whether to continue or finalize. For example, once stock prices are fetched, the agent can compute trends with `calculate()` rather than guessing them.

---

## IV. Future Improvements (5 Points)

*How would you scale this for a production-level AI agent system?*

- **Scalability**: Introduce an asynchronous orchestration layer for tool execution, use a task queue for long-running tools, and implement caching for repeated stock queries.
- **Safety**: Add a supervisor or validator layer to audit model actions before execution, sanitize tool inputs, and require the model to justify each tool call in the `Thought` block.
- **Performance**: Use a retrieval-augmented design with a vector database for tool descriptions and user context, and cache heatmap results for common stock tickers to reduce latency.

---

> [!NOTE]
> Submit this report by renaming it to `REPORT_[YOUR_NAME].md` and placing it in this folder.
