from typing import Dict, Optional

from tools.base_tool import BaseTool, ToolRegistry
from tools.utils import normalize_tool_schema, summarize_previous


@ToolRegistry.register
class SemanticDissector(BaseTool):
    TOOL_NAME = "semantic_dissector"
    DESCRIPTION = "Analyze literal and implied meanings in text."

    def analyze(
        self,
        query_text: str,
        image_embedding,
        retrieved_context: Dict,
        previous_outputs: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        _ = image_embedding
        retrieved_texts = retrieved_context.get("retrieved_texts") or []
        prev_summary = summarize_previous(previous_outputs)

        system_prompt = (
            "You are the Semantic Dissector tool for harmful meme detection. "
            "Analyze literal meaning, implied meaning, target of the joke, speaker stance, slang, coded wording, "
            "puns, double meanings, and whether the text itself degrades, threatens, objectifies, or stereotypes someone. "
            "Return JSON using the shared Tool-meme tool schema."
        )
        user_prompt = (
            "Task: Identify the target, implied punchline, and semantic harm signal. "
            "If the wording is merely sensitive but not attacking anyone, mark that as harmless or unknown.\n\n"
            f"Query Text:\n{query_text}\n\n"
            "Retrieved Texts (top 3):\n" + "\n".join(retrieved_texts[:3]) + "\n\n"
            f"Previous tool outputs:\n{prev_summary}\n\n"
            "Return JSON only."
        )

        result = self._run_llm(system_prompt, user_prompt)
        return normalize_tool_schema(result, self.TOOL_NAME, "Semantic analysis completed.")
