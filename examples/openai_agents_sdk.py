"""
Example: DeepCQ with the OpenAI Agents SDK.

The pattern is simple: use deepcq to build a compact, query-specific
context block, then hand that block to the agent SDK as part of the turn.
"""
from __future__ import annotations

from deepcq import DeepCQ, MemorySchema


def build_cq() -> DeepCQ:
    schema = MemorySchema(
        sections=["status", "errors", "metrics", "actions"],
        confidence_threshold=0.25,
    )
    cq = DeepCQ(schema=schema)

    cq.state.set("status", "[status]\nAll services running\n")
    cq.state.set("errors", "[errors]\n- OOM at 02:15\n")
    cq.state.set("metrics", "[metrics]\nCPU: 78%\nMemory: 4.2GB\n")
    cq.state.set("actions", "[actions]\nScale the cache tier before the next release.\n")

    return cq


def main() -> None:
    try:
        from agents import Agent, Runner
    except ImportError as exc:
        raise SystemExit(
            "Install the OpenAI Agents SDK first (for example: pip install openai-agents)."
        ) from exc

    cq = build_cq()
    user_message = "Why is the service under pressure?"
    memory_context = cq.query(user_message).content

    agent = Agent(
        name="deepcq-ops-assistant",
        instructions=(
            "Answer using the provided memory context. "
            "If the context does not contain the answer, say so explicitly."
        ),
    )

    result = Runner.run_sync(
        agent,
        input=(
            "Relevant memory from deepcq:\n"
            f"{memory_context}\n\n"
            f"User request: {user_message}"
        ),
    )

    print(getattr(result, "final_output", result))


if __name__ == "__main__":
    main()