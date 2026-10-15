"""
Example: DeepCQ with the Claude Agent SDK.

This keeps deepcq on the retrieval side and lets the Claude agent runtime
handle the reasoning and response generation.
"""
from __future__ import annotations

from deepcq import DeepCQ, MemorySchema


def build_cq() -> DeepCQ:
    schema = MemorySchema(
        sections=["customer", "incident", "sla", "next-steps"],
        confidence_threshold=0.25,
    )
    cq = DeepCQ(schema=schema)

    cq.state.set("customer", "[customer]\nName: Northstar Retail\nPlan: Enterprise\n")
    cq.state.set("incident", "[incident]\n- Checkout latency rose above 2s\n- 503s on payment retries\n")
    cq.state.set("sla", "[sla]\nP1 response target: 15 minutes\nResolution target: 4 hours\n")
    cq.state.set("next-steps", "[next-steps]\nEscalate to payments and platform on-call.\n")

    return cq


def main() -> None:
    try:
        from claude_agent_sdk import Agent, Runner
    except ImportError as exc:
        raise SystemExit(
            "Install the Claude Agent SDK first (for example: pip install claude-agent-sdk)."
        ) from exc

    cq = build_cq()
    user_message = "Summarize the incident and give the next step."
    memory_context = cq.query(user_message).content

    agent = Agent(
        name="deepcq-support-assistant",
        instructions=(
            "Use the retrieved context before answering. "
            "Keep the response operational and short."
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