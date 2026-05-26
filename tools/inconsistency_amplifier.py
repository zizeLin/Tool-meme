from typing import Dict, Optional

from tools.base_tool import BaseTool, ToolRegistry
from tools.utils import normalize_tool_schema, summarize_previous


@ToolRegistry.register
class InconsistencyAmplifier(BaseTool):
    TOOL_NAME = "inconsistency_amplifier"
    DESCRIPTION = "Quantify inconsistencies across retrieved labels and similarity."

    def analyze(
        self,
        query_text: str,
        image_embedding,
        retrieved_context: Dict,
        previous_outputs: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        _ = image_embedding
        labels = retrieved_context.get("labels") or []
        scores = retrieved_context.get("scores") or []
        prev_summary = summarize_previous(previous_outputs)
        label_hint = f"labels={labels[:5]}" if labels else "labels=none"
        similarity_hint = f"avg_similarity={sum(scores)/len(scores):.3f}" if scores else "avg_similarity=unknown"

        system_prompt = (
            "You are the Inconsistency Amplifier tool for harmful meme detection. "
            "Identify contradictions among the target reading, image-text alignment, retrieved labels, and previous tool judgments. "
            "Use contradictions to lower confidence or surface ambiguity; do not convert ambiguity into harm by default. "
            "Return JSON using the shared Tool-meme tool schema."
        )
        user_prompt = (
            "Task: Identify inconsistencies that could flip the final harmful/harmless label or require caution.\n\n"
            f"Query Text:\n{query_text}\n\n"
            f"Label hint: {label_hint}\n"
            f"Similarity hint: {similarity_hint}\n\n"
            f"Previous tool outputs:\n{prev_summary}\n\n"
            "Return JSON only."
        )

        result = self._run_llm(system_prompt, user_prompt)
        return normalize_tool_schema(result, self.TOOL_NAME, "Inconsistency analysis completed.")
