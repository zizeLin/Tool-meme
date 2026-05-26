from typing import Dict, Optional

from tools.base_tool import BaseTool, ToolRegistry
from tools.utils import normalize_tool_schema, summarize_previous


@ToolRegistry.register
class EvidenceComparator(BaseTool):
    TOOL_NAME = "evidence_comparator"
    DESCRIPTION = "Compare retrieved label distribution as evidence strength."

    def analyze(
        self,
        query_text: str,
        image_embedding,
        retrieved_context: Dict,
        previous_outputs: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        _ = image_embedding
        labels = retrieved_context.get("labels") or []
        prev_summary = summarize_previous(previous_outputs)

        system_prompt = (
            "You are the Evidence Comparator tool for harmful meme detection. "
            "Compare retrieved label distribution, semantic closeness, and prior reliability. "
            "Do not blindly follow majority labels; decide whether retrieval provides strong, weak, or conflicting calibration. "
            "Return JSON using the shared Tool-meme tool schema."
        )
        user_prompt = (
            "Task: Compare retrieved labels against target-specific evidence from previous tools. "
            "Return harmful/harmless only when retrieval is close and consistent; otherwise return unknown.\n\n"
            f"Query Text:\n{query_text}\n\n"
            f"Retrieved Labels (top 10):\n{labels[:10]}\n\n"
            f"Previous tool outputs:\n{prev_summary}\n\n"
            "Return JSON only."
        )

        result = self._run_llm(system_prompt, user_prompt)
        return normalize_tool_schema(result, self.TOOL_NAME, "Evidence comparison completed.")
