"""
Adapters making DeepCQ usable as a memory/context backend for agent frameworks.

ToolContextProvider  — formats QueryResult as a prompt-injectable context string.
LangChainMemoryAdapter — duck-types LangChain's BaseChatMemory interface.

Neither adapter imports LangChain or any other framework. They implement the
interfaces by duck-typing so they work without installing any extras.
"""
from __future__ import annotations

import logging

from deepcq import DeepCQ
from .prompt_library import PromptLibrary

logger = logging.getLogger("deepcq")


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

    def get_context_with_note(self, query: str, reload_tool: str = "get_section") -> str:
        """
        Filtered memory content plus a line naming the sections left out and
        how to load one — so the model neither assumes that data is missing
        nor re-fetches it from elsewhere. Pair with section_tool_spec() and
        get_section() to give the agent the reload tool the note refers to.
        """
        result = self._cq.query(query)
        note = result.omitted_note(f'Load one with {reload_tool}(name="<section>").')
        return f"{result.content}\n\n{note}" if note else result.content

    def get_section(self, name: str) -> str:
        """Tool handler for section_tool_spec(): one section's stored text."""
        text = self._cq.get_section(name)
        if text is not None:
            return text
        names = [s for s in self._cq.schema.sections]
        return f"No section named {name!r}. Sections: {', '.join(names)}"

    def section_tool_spec(self, name: str = "get_section") -> dict:
        """
        Function-calling definition (name / description / JSON-schema
        parameters) for a tool that loads one omitted section via
        get_section(). Rename the parameters key to `input_schema`
        (Anthropic) or wrap as {"type": "function", "function": ...} (OpenAI).
        """
        return {
            "name": name,
            "description": (
                "Load one memory section that was omitted from the context as not "
                "relevant. Matches the section name case-insensitively, or by prefix. "
                f"Sections: {', '.join(self._cq.schema.sections)}."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Section name or prefix"}},
                "required": ["name"],
            },
        }

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
        if query:
            return {self.memory_key: self._cq.query(query).content}
        total = self._cq._count_fallback("no_query")
        logger.warning(
            "deepcq: load_memory_variables got no input; returning the full history "
            "unfiltered (process total no_query: %d)", total,
        )
        return {self.memory_key: self._cq.history.as_text()}

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
