from typing import Dict, Optional

from tools.base_tool import BaseTool, ToolRegistry
from tools.utils import normalize_tool_schema, summarize_previous


@ToolRegistry.register
class KnowledgeGrounding(BaseTool):
    TOOL_NAME = "knowledge_grounding"
    DESCRIPTION = "Check if query text aligns with retrieved context."

    def analyze(
        self,
        query_text: str,
        image_embedding,
        retrieved_context: Dict,
        previous_outputs: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        _ = image_embedding
        retrieved_texts = retrieved_context.get("retrieved_texts") or []
        event_contexts = retrieved_context.get("event_contexts") or []
        prev_summary = summarize_previous(previous_outputs)

        system_prompt = (
            "You are the Knowledge Grounding tool for harmful meme detection. "
            "Verify whether public events, historical references, named groups, or social context affect the harm label. "
            "Sensitive events are harmful only when the meme mocks victims, endorses harm, spreads hostile stereotypes, or uses tragedy as abuse. "
            "Return JSON using the shared Tool-meme tool schema."
        )
        user_prompt = (
            "Task: Check factual/event context and whether it supports harmful, harmless, or unknown classification.\n\n"
            f"Query Text:\n{query_text}\n\n"
            "Retrieved Texts (top 3):\n" + "\n".join(retrieved_texts[:3]) + "\n\n"
            "Event Contexts:\n" + "; ".join(event_contexts[:3]) + "\n\n"
            f"Previous tool outputs:\n{prev_summary}\n\n"
            "Return JSON only."
        )

        result = self._run_llm(system_prompt, user_prompt)
        return normalize_tool_schema(result, self.TOOL_NAME, "Knowledge grounding completed.")
