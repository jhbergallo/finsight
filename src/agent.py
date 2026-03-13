import json
from typing import List, Dict, Any

from openai import OpenAI

from src.vector_store import VectorStore
from src.config import settings

SYSTEM_PROMPT = """You are FinSight, an expert financial analyst assistant.
You have access to a knowledge base of financial documents (earnings reports, 10-Ks, analyst notes, etc.).

Your job is to answer questions accurately and concisely, grounding your responses in the retrieved context.

Guidelines:
- Always base your answers on the provided context. If the context is insufficient, say so clearly.
- When referencing specific figures (revenue, margins, EPS, etc.), be precise.
- If comparing across documents, note which document each figure comes from.
- Do not fabricate numbers or make up information not present in the context.
- If a question is outside the scope of the loaded documents, say so instead of guessing.
- Keep answers structured and easy to scan when the question involves multiple data points.
"""

RETRIEVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the financial document knowledge base for relevant context. "
            "Use this before answering any question that requires specific information "
            "from the uploaded documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to retrieve relevant document chunks.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve. Default is 6.",
                    "default": 6,
                },
            },
            "required": ["query"],
        },
    },
}


class FinancialAgent:
    """
    ReAct-style agent that decides when to retrieve context before answering.
    Uses tool-calling to interact with the vector store, enabling multi-step
    reasoning over financial documents.
    """

    def __init__(self, vector_store: VectorStore, api_key: str = None):
        self.vs = vector_store
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = settings.llm_model

    def run(self, query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        messages = self._build_messages(query, chat_history or [])
        sources_used = []

        # Agentic loop — allows multi-step tool use
        for _ in range(5):  # max iterations
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[RETRIEVAL_TOOL],
                tool_choice="auto",
                temperature=settings.temperature,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg)

                for tool_call in msg.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    results = self.vs.search(
                        query=args["query"],
                        top_k=args.get("top_k", settings.top_k_retrieval),
                    )

                    sources_used.extend(results)

                    tool_result = self._format_context(results)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )
            else:
                # Final answer
                answer = msg.content or "I couldn't generate a response."
                unique_sources = self._deduplicate_sources(sources_used)
                return {"answer": answer, "sources": unique_sources}

        return {
            "answer": "I reached the maximum reasoning steps. Please try rephrasing your question.",
            "sources": [],
        }

    def _build_messages(self, query: str, history: List[Dict]) -> List[Dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in history[-6:]:  # limit context window to last 6 turns
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})
        return messages

    def _format_context(self, results: List[Dict]) -> str:
        if not results:
            return "No relevant context found in the knowledge base."

        parts = []
        for r in results:
            parts.append(
                f"[Source: {r['source']} | Chunk {r['chunk_id']} | Score: {r['score']:.2f}]\n{r['content']}"
            )

        return "\n\n---\n\n".join(parts)

    def _deduplicate_sources(self, sources: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for s in sources:
            key = (s["source"], s["chunk_id"])
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique
