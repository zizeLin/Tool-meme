from typing import Dict, Optional

from tools.base_tool import BaseTool, ToolRegistry
from tools.utils import clamp, normalize_tool_schema, summarize_previous


@ToolRegistry.register
class CrossModalAligner(BaseTool):
    TOOL_NAME = "cross_modal_aligner"
    DESCRIPTION = "Estimate image-text alignment using retrieval similarity as a proxy."

    def analyze(
        self,
        query_text: str,
        image_embedding,
        retrieved_context: Dict,
        previous_outputs: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        alignment_score = retrieved_context.get("alignment_score")
        if alignment_score is not None:
            return normalize_tool_schema(
                {
                    "trace": f"Used offline alignment score: {alignment_score:.3f}.",
                    "conf": clamp((alignment_score + 1.0) / 2.0),
                    "pred": "unknown",
                    "evidence": [f"offline alignment score={alignment_score:.3f}"],
                    "status": "valid",
                },
                self.TOOL_NAME,
                "Cross-modal alignment estimated.",
            )

        scores = retrieved_context.get("scores") or []
        retrieved_texts = retrieved_context.get("retrieved_texts") or []
        prev_summary = summarize_previous(previous_outputs)
        similarity_hint = f"avg_similarity={sum(scores)/len(scores):.3f}" if scores else "avg_similarity=unknown"
        emb_hint = "image_embedding=none"
        if image_embedding is not None:
            emb_hint = "image_embedding=provided"

        system_prompt = (
            "You are the Cross-Modal Aligner tool for harmful meme detection. "
            "Assess whether the image changes, reinforces, or contradicts the text's harmful meaning. "
            "Focus on visual target identity, depicted action, facial expression, object symbolism, and image-text mismatch. "
            "Return JSON using the shared Tool-meme tool schema."
        )
        user_prompt = (
            "Task: Assess whether image-text alignment strengthens a harmful reading, weakens it, or is inconclusive.\n\n"
            f"Query Text:\n{query_text}\n\n"
            "Retrieved Texts (top 3):\n" + "\n".join(retrieved_texts[:3]) + "\n\n"
            f"Similarity hint: {similarity_hint}\n\n"
            f"Image embedding hint: {emb_hint}\n\n"
            f"Previous tool outputs:\n{prev_summary}\n\n"
            "Return JSON only."
        )

        result = self._run_llm(system_prompt, user_prompt)
        if image_embedding is None:
            result["trace"] = (
                result.get("trace")
                or result.get("reasoning_trace")
                or "Cross-modal alignment estimated."
            ) + " image_embedding is None; alignment uses textual hints."
        return normalize_tool_schema(result, self.TOOL_NAME, "Cross-modal alignment estimated.")
