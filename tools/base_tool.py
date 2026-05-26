import json
from typing import Dict, Optional, Type

from utils.openai_client import OpenAIModelConfig, get_openai_response_with_parts
from utils.prompting import HARMFUL_MEME_RUBRIC, JSON_ONLY_RULE, TOOL_DECISION_GUIDANCE


TOOL_OUTPUT_SCHEMA = (
    "Return ONLY JSON with fields: "
    "{\"name\": \"tool_name\", \"trace\": \"concise evidence trace\", "
    "\"pred\": \"harmful|harmless|unknown\", \"conf\": 0.0, "
    "\"evidence\": [\"short evidence item\"], \"status\": \"valid|abstain|failed\"}. "
    "Use harmful/harmless/unknown as the only prediction words."
)


class BaseTool:
    """
    Base interface for tools.
    Input: (query_text, image_embedding, retrieved_context)
    Output: shared Tool-meme tool schema plus legacy compatibility aliases.
    """

    TOOL_NAME: str = ""
    DESCRIPTION: str = ""

    def analyze(
        self,
        query_text: str,
        image_embedding,
        retrieved_context: Dict,
        previous_outputs: Optional[Dict[str, Dict]] = None,
    ) -> Dict:
        raise NotImplementedError

    def _run_llm(self, system_prompt: str, user_prompt: str) -> Dict:
        parts = [
            {"type": "text", "text": (
                f"{system_prompt}\n\n"
                f"{HARMFUL_MEME_RUBRIC}\n\n"
                f"{TOOL_DECISION_GUIDANCE}\n\n"
                f"{TOOL_OUTPUT_SCHEMA}\n"
                f"{JSON_ONLY_RULE}"
            )},
            {"type": "text", "text": user_prompt},
        ]
        cfg = OpenAIModelConfig.from_env()
        raw = get_openai_response_with_parts(parts, cfg)
        return _parse_json(raw)


def _parse_json(text: str) -> Dict:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        return {}


class ToolRegistry:
    _tools: Dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls, tool_class: Type[BaseTool]):
        if tool_class.TOOL_NAME:
            cls._tools[tool_class.TOOL_NAME] = tool_class
        return tool_class

    @classmethod
    def create(cls, tool_name: str) -> Optional[BaseTool]:
        tool_class = cls._tools.get(tool_name)
        if tool_class:
            return tool_class()
        return None
