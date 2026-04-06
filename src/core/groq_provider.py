import time
from typing import Dict, Any, Optional, Generator
from openai import OpenAI
from src.core.llm_provider import LLMProvider
import os

class GroqProvider(LLMProvider):
    """
    LLM Provider for Groq's ultra-fast inference API.
    Uses the OpenAI-compatible SDK.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "groq",
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set in .env")
        exit(1)

    provider = GroqProvider(api_key=api_key)
    prompt = "Explain what an AI Agent is in one sentence."

    print(f"--- Testing Groq Provider ({provider.model_name}) ---")
    print(f"User: {prompt}\n")

    # Test generate()
    result = provider.generate(prompt)
    print(f"Assistant: {result['content']}")
    print(f"\nTokens: {result['usage']}")
    print(f"Latency: {result['latency_ms']}ms")

    # Test stream()
    print(f"\n--- Streaming test ---")
    print("Assistant: ", end="", flush=True)
    for chunk in provider.stream(prompt):
        print(chunk, end="", flush=True)
    print("\n\nGroq Provider is working correctly!")