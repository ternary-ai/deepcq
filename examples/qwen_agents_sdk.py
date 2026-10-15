"""
Example: DeepCQ with the Qwen Agent SDK.

The main idea is the same as the other SDK examples: retrieve a compact
context block from deepcq, then hand it to the agent as preloaded context.
"""
from __future__ import annotations

from deepcq import DeepCQ, MemorySchema


def build_cq() -> DeepCQ:
    schema = MemorySchema(
        sections=["research", "risk", "plan", "open-questions"],
        confidence_threshold=0.25,
    )
    cq = DeepCQ(schema=schema)

    cq.state.set("research", "[research]\nTicker: AAPL\nTheme: consumer device refresh\n")
    cq.state.set("risk", "[risk]\n- Supply chain delay for one component\n- Margin pressure from promotions\n")
    cq.state.set("plan", "[plan]\nDraft the launch summary and flag any unresolved blockers.\n")
    cq.state.set("open-questions", "[open-questions]\nConfirm launch date with the product owner.\n")

    return cq


def main() -> None:
    try:
        from qwen_agent.agents import Assistant
    except ImportError as exc:
        raise SystemExit(
            "Install the Qwen Agent SDK first (for example: pip install qwen-agent)."
        ) from exc

    cq = build_cq()
    user_message = "What should I tell the team about launch risk?"
    memory_context = cq.query(user_message).content

    assistant = Assistant(
        llm={"model": "qwen-max"},
        system_message=(
            "Use the retrieved deepcq context as the primary source of truth."
        ),
    )

    result = assistant.run(
        [
            {"role": "system", "content": f"DeepCQ context:\n{memory_context}"},
            {"role": "user", "content": user_message},
        ]
    )

    print(result)


if __name__ == "__main__":
    main()