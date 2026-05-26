from typing import Dict, Optional

from tools.base_tool import BaseTool, ToolRegistry
from tools.utils import normalize_tool_schema, summarize_previous


@ToolRegistry.register
class CulturalDecoder(BaseTool):
    TOOL_NAME = "cultural_decoder"
    DESCRIPTION = "Detect subculture symbols or cultural references."

    CULTURE_MARKERS = [
        "pepe",
        "wojak",
        "doomer",
        "boomer",
        "kek",
        "npc",
        "chad",
        "soyboy",
        "meme",
        "troll",
        "rickroll",
        "doge",
    ]

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
            "You are the Cultural Decoder tool for harmful meme detection. "
            "Identify meme templates, dog whistles, coded hate, political/religious/cultural references, and in-group slang. "
            "Explain whether the cultural reference turns a group, victim, or tragedy into the object of contempt. "
            "Return JSON using the shared Tool-meme tool schema."
        )
        user_prompt = (
            "Task: Decode cultural/subculture references and judge whether they support a harmful or benign reading.\n\n"
            f"Query Text:\n{query_text}\n\n"
            "Retrieved Texts (top 3):\n" + "\n".join(retrieved_texts[:3]) + "\n\n"
            f"Previous tool outputs:\n{prev_summary}\n\n"
            "Return JSON only."
        )

        result = self._run_llm(system_prompt, user_prompt)
        return normalize_tool_schema(result, self.TOOL_NAME, "Cultural decoding completed.")
