import os
import re
from typing import List, Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger

class ReActAgent:
    """
    SKELETON: A ReAct-style Agent that follows the Thought-Action-Observation loop.
    Students should implement the core loop logic and tool execution.
    """
    
    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 5, verbose: bool = False):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        self.history = []

    def get_system_prompt(self) -> str:
        """
        TODO: Implement the system prompt that instructs the agent to follow ReAct.
        Should include:
        1.  Available tools and their descriptions.
        2.  Format instructions: Thought, Action, Observation.
        """
        tool_descriptions = "\n".join([f"- {t['name']}: {t['description']}" for t in self.tools])
        return f"""You are an intelligent assistant. You have access to the following tools:
{tool_descriptions}

STRICT RULES:
1. You MUST call a tool using Action BEFORE giving a Final Answer if the question requires real-time data, stock prices, or any external information.
2. NEVER fabricate, assume, or hallucinate data. If you need data, you MUST use a tool to get it.
3. Only give a Final Answer AFTER you have received real Observation data from tool calls.
4. You may call multiple tools in sequence (one per step).

Use EXACTLY this format (no extra text before Thought):
Thought: <your reasoning about what to do next>
Action: tool_name(<arguments>)

After you receive an Observation, continue with another Thought.
When you have enough real data to answer, use:
Thought: <your final reasoning>
Final Answer: <your answer based on real Observation data>"""

    def run(self, user_input: str) -> str:
        """
        ReAct loop: Thought -> Action -> Observation, repeat until Final Answer.
        """
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})

        current_prompt = user_input
        steps = 0

        while steps < self.max_steps:
            # 1. Generate LLM response
            result = self.llm.generate(current_prompt, system_prompt=self.get_system_prompt())
            response_text = result["content"]
            logger.log_event("LLM_RESPONSE", {"step": steps + 1, "response": response_text})

            # Print response_text each time LLM responds
            print(f"\n{'='*60}")
            print(f"Step {steps + 1}")
            print(f"{'='*60}")
            print(response_text)

            # # Print Thought if verbose
            # if self.verbose:
            #     thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", response_text, re.DOTALL)
            #     if thought_match:
            #         print(f"\n{'='*60}")
            #         print(f"Step {steps + 1}")
            #         print(f"{'='*60}")
            #         print(f"THOUGHT: {thought_match.group(1).strip()}")

            # 2. Check for Final Answer
            final_match = re.search(r"Final Answer:\s*(.*)", response_text, re.DOTALL)
            if final_match:
                answer = final_match.group(1).strip()
                # if self.verbose:
                #     print(f"FINAL ANSWER: {answer}")
                logger.log_event("AGENT_END", {"steps": steps + 1, "status": "final_answer"})
                return answer

            # 3. Parse Action
            action_match = re.search(r"Action:\s*(\w+)\((.*)?\)", response_text, re.DOTALL)
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_args = (action_match.group(2) or "").strip()
                # if self.verbose:
                #     print(f"ACTION: {tool_name}({tool_args})")
                logger.log_event("TOOL_CALL", {"tool": tool_name, "args": tool_args})

                # 4. Execute tool and get Observation
                observation = self._execute_tool(tool_name, tool_args)
                if self.verbose:
                    print(f"OBSERVATION: {observation}")
                logger.log_event("TOOL_RESULT", {"tool": tool_name, "result": observation})

                # 5. Append the full exchange to the prompt for the next iteration
                current_prompt = (
                    f"{current_prompt}\n\n"
                    f"{response_text}\n"
                    f"Observation: {observation}\n"
                )
            else:
                # No Action and no Final Answer — treat the response as the final answer
                logger.log_event("AGENT_END", {"steps": steps + 1, "status": "no_action"})
                return response_text.strip()

            steps += 1

        logger.log_event("AGENT_END", {"steps": steps, "status": "max_steps"})
        return "Reached maximum steps without a final answer."

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """
        Execute a tool by name, passing args as a string.
        """
        for tool in self.tools:
            if tool['name'] == tool_name:
                try:
                    return tool['function'](args)
                except Exception as e:
                    return f"Error executing {tool_name}: {str(e)}"
        return f"Tool '{tool_name}' not found. Available: {[t['name'] for t in self.tools]}"


