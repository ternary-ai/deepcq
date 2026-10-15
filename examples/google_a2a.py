"""
Example: DeepCQ inside a Google A2A task handler.

Google A2A is protocol-first, so this example keeps the protocol boundary
thin: a task arrives, deepcq filters memory for that task, and the handler
returns a response payload.
"""
from __future__ import annotations

from deepcq import DeepCQ, MemorySchema


def build_cq() -> DeepCQ:
    schema = MemorySchema(
        sections=["project", "incident", "timeline", "follow-up"],
        confidence_threshold=0.25,
    )
    cq = DeepCQ(schema=schema)

    cq.state.set("project", "[project]\nAgent-to-agent bridge for support escalation\n")
    cq.state.set("incident", "[incident]\nA downstream service dropped messages during deploy\n")
    cq.state.set("timeline", "[timeline]\n14:05 deploy started\n14:17 first error spike\n14:29 rollback completed\n")
    cq.state.set("follow-up", "[follow-up]\nCollect logs and confirm whether any tasks were lost\n")

    return cq


def handle_a2a_task(task: dict[str, str]) -> dict[str, str]:
    """Protocol boundary: adapt a task payload into a deepcq-backed reply."""

    cq = build_cq()
    user_message = task["message"]
    memory_context = cq.query(user_message).content

    return {
        "task_id": task["task_id"],
        "status": "completed",
        "response": (
            "Context for the agent:\n"
            f"{memory_context}\n\n"
            "Recommended next step: verify whether the lost messages can be replayed."
        ),
    }


def main() -> None:
    sample_task = {
        "task_id": "a2a-001",
        "message": "Do we need to alert the on-call team about message loss?",
    }
    result = handle_a2a_task(sample_task)
    print(result["response"])


if __name__ == "__main__":
    main()