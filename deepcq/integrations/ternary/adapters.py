"""
Adapters making DeepCQ usable as a memory/context backend for agent frameworks.

ToolContextProvider  — formats QueryResult as a prompt-injectable context string.
LangChainMemoryAdapter — duck-types LangChain's BaseChatMemory interface.

Neither adapter imports LangChain or any other framework. They implement the
interfaces by duck-typing so they work without installing any extras.
"""
from __future__ import annotations

from deepcq import DeepCQ
from .prompt_library import PromptLibrary


class ToolContextProvider:
    """
    Formats DeepCQ query results as context strings for LLM prompts.

    Matches the pattern used in ternary-ai/agents OpenAI Financial
    Researcher Agent notebook: retrieve relevant context, inject into
    the system prompt or tool schema before calling the LLM.

    Example::

        cq = DeepCQ(schema=schema)
        provider = ToolContextProvider(cq)

        # Before each LLM call:
        context = provider.get_context("What is portfolio concentration risk?")
        prompt = f"Use this context:\\n{context}\\n\\nQuestion: ..."
    """

    def __init__(self, cq: DeepCQ) -> None:
        self._cq = cq

    def get_context(self, query: str) -> str:
        """Return filtered memory content relevant to the query."""
        return self._cq.query(query).content

    def get_prompt_template(self, query: str, library: PromptLibrary) -> str:
        """
        Retrieve the best matching prompt template(s) from a PromptLibrary
        for the given query. Returns the raw template text with {variable}
        placeholders intact — the caller fills them in.
        """
        return library.query(query).content

    def get_context_with_template(
        self, query: str, library: PromptLibrary
    ) -> tuple[str, str]:
        """
        Convenience: return (memory_context, prompt_template) together.

        Useful when you want to inject both retrieved state and a prompt
        template into an LLM call in one step.
        """
        return self.get_context(query), self.get_prompt_template(query, library)


class LangChainMemoryAdapter:
    """
    Makes DeepCQ usable as a LangChain chat memory component.

    Duck-types the LangChain BaseChatMemory interface (save_context /
    load_memory_variables / clear) without importing LangChain itself.
    Drop it into any LangChain chain that accepts a memory= argument.

    The adapter stores conversation turns in DeepCQ's History tier (T3)
    and caches tool results in the SessionCache tier (T2). On each call to
    load_memory_variables, it runs a deepcq query against the incoming
    message to return only the most relevant prior context.

    Example (LangChain)::

        cq = DeepCQ(schema=schema)
        memory = LangChainMemoryAdapter(cq)

        chain = ConversationalRetrievalChain(
            llm=llm,
            retriever=retriever,
            memory=memory,
        )

    Example (manual / OpenAI Agents SDK)::

        memory.save_context({"input": user_msg}, {"output": agent_reply})
        ctx = memory.load_memory_variables({"input": next_user_msg})
        print(ctx["history"])  # filtered relevant context
    """

    memory_key: str = "history"

    def __init__(self, cq: DeepCQ) -> None:
        self._cq = cq

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Store one conversation turn (human + AI) in DeepCQ History."""
        human = inputs.get("input") or inputs.get("human_input") or ""
        ai = outputs.get("output") or outputs.get("response") or ""
        if human:
            self._cq.history.add("human", human)
        if ai:
            self._cq.history.add("ai", ai)

    def load_memory_variables(self, inputs: dict) -> dict[str, str]:
        """
        Return filtered context relevant to the incoming message.

        LangChain calls this before each LLM invocation. The returned dict
        must have the key matching memory_key (default "history").
        """
        query = inputs.get("input") or inputs.get("human_input") or ""
        result = self._cq.query(query) if query else None
        content = result.content if result else self._cq.history.as_text()
        return {self.memory_key: content}

    def clear(self) -> None:
        """Reset History and SessionCache (analogous to LangChain memory.clear())."""
        self._cq.history.clear()
        self._cq.cache.clear()

    # LangChain also accesses these attributes on memory objects
    @property
    def chat_memory(self):
        """Compatibility shim — returns an object with a messages property."""
        turns = self._cq.history.turns()

        class _Messages:
            def __init__(self, turns):
                self.messages = [
                    {"role": t.role, "content": t.text} for t in turns
                ]

        return _Messages(turns)

    @property
    def output_key(self) -> str:
        return "output"

    @property
    def input_key(self) -> str:
        return "input"
