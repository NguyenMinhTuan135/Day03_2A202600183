"""
Chatbot Baseline: Simple LLM chatbot WITHOUT tools.
Demonstrates limitations when faced with multi-step or real-time data questions.
"""
import os
from dotenv import load_dotenv
from src.core.groq_provider import GroqProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


def chatbot(llm: GroqProvider, question: str) -> str:
    system_prompt = (
        "You are a helpful Vietnamese stock market assistant. "
        "Answer the user's question directly. Be concise."
    )

    logger.log_event("CHATBOT_START", {"question": question, "model": llm.model_name})

    result = llm.generate(question, system_prompt=system_prompt)

    tracker.track_request(
        provider=result["provider"],
        model=llm.model_name,
        usage=result["usage"],
        latency_ms=result["latency_ms"],
    )

    logger.log_event("CHATBOT_END", {
        "question": question,
        "latency_ms": result["latency_ms"],
        "tokens": result["usage"],
    })

    return result["content"]


# ── Test Cases ─────────────────────────────────────────────────

TEST_CASES = [
    # Case 1: Simple knowledge question — chatbot CAN answer
    {
        "label": "Simple Q (no tool needed)",
        "question": "What is FPT company?",
        "expect": "Chatbot should answer correctly from its training knowledge.",
    },
    # Case 2: Real-time data — chatbot CANNOT answer (no tool)
    {
        "label": "Real-time price (needs tool)",
        "question": "How much is the close price of FPT today",
        "expect": "Chatbot will refuse or HALLUCINATE a price.",
    },
    # Case 3: Multi-step reasoning — chatbot CANNOT answer
    {
        "label": "Multi-step comparison (needs 2 tools + calculation)",
        "question": "Compare the stock price of FPT and VIC today?",
        "expect": "Chatbot will hallucinate numbers or say it cannot access real-time data.",
    },
    # Case 4: Calculation on real data — chatbot CANNOT answer
    {
        "label": "Calculation on live data (needs tool + math)",
        "question": "Caculate the average stock price of VIC in 5 latest sessions?",
        "expect": "Chatbot will make up numbers since it has no tool to fetch data.",
    },
]


if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not set in .env")
        exit(1)

    llm = GroqProvider(api_key=api_key)

    print("=" * 60)
    print("CHATBOT BASELINE TEST")
    print(f"Model: {llm.model_name} (no tools)")
    print("=" * 60)

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n--- Case {i}: {case['label']} ---")
        print(f"Question: {case['question']}")
        print(f"Expected: {case['expect']}")
        print()

        answer = chatbot(llm, case["question"])

        print(f"Chatbot:  {answer}")
        print("-" * 60)
