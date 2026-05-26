from typing import Dict, Optional

from tools.base_tool import BaseTool, ToolRegistry
from tools.utils import normalize_tool_schema, summarize_previous


@ToolRegistry.register
class RhetoricalScanner(BaseTool):
    TOOL_NAME = "rhetorical_scanner"
    DESCRIPTION = "Detect sarcasm, irony, exaggeration, metaphor."

    MARKERS = [
        "/s",
        "yeah right",
        "sure",
        "totally",
        "as if",
        "literally",
        "obviously",
        "clearly",
        "right...",
        "lol",
        "lmao",
        "jk",
    ]

    def analyze(
        self,
        query_text: str,
        image_embedding,
        retrieved_context: Dict,
        previous_outputs: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        _ = image_embedding
        prev_summary = summarize_previous(previous_outputs)

        system_prompt = (
            "You are the Rhetorical Scanner tool for harmful meme detection. "
            "Identify irony, sarcasm, exaggeration, metaphor, reversal, euphemism, quotation, and dark-humor framing. "
            "Determine whether the rhetoric makes hatred, humiliation, abuse, or a protected-group stereotype into the joke. "
            "Return JSON using the shared Tool-meme tool schema."
        )
        user_prompt = (
            "Task: Detect rhetorical devices and infer whether they invert the literal meaning toward harmful or harmless intent.\n\n"
            f"Query Text:\n{query_text}\n\n"
            f"Previous tool outputs:\n{prev_summary}\n\n"
            "Return JSON only."
        )

        result = self._run_llm(system_prompt, user_prompt)
        return normalize_tool_schema(result, self.TOOL_NAME, "Rhetorical analysis completed.")
