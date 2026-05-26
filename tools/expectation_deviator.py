from typing import Dict, Optional

from tools.base_tool import BaseTool, ToolRegistry
from tools.utils import normalize_tool_schema, summarize_previous


@ToolRegistry.register
class ExpectationDeviator(BaseTool):
    TOOL_NAME = "expectation_deviator"
    DESCRIPTION = "Estimate deviation between query and retrieved context."

    def analyze(
        self,
        query_text: str,
        image_embedding,
        retrieved_context: Dict,
        previous_outputs: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        _ = image_embedding
        scores = retrieved_context.get("scores") or []
        prev_summary = summarize_previous(previous_outputs)
        similarity_hint = f"avg_similarity={sum(scores)/len(scores):.3f}" if scores else "avg_similarity=unknown"

        system_prompt = (
            "You are the Expectation Deviator tool for harmful meme detection. "
            "Estimate whether expectation violation between benign visuals/context and hostile text creates the harmful punchline. "
            "Distinguish harmless absurdity from shock humor that humiliates, threatens, or stereotypes a target. "
            "Return JSON using the shared Tool-meme tool schema."
        )
        user_prompt = (
            "Task: Evaluate expectation deviation and whether the surprise itself creates harm.\n\n"
            f"Query Text:\n{query_text}\n\n"
            f"Similarity hint: {similarity_hint}\n\n"
            f"Previous tool outputs:\n{prev_summary}\n\n"
            "Return JSON only."
        )

        result = self._run_llm(system_prompt, user_prompt)
        return normalize_tool_schema(result, self.TOOL_NAME, "Expectation deviation estimated.")
