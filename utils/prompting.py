"""Prompt fragments shared by GPT-4o based Tool-meme modules."""

HARMFUL_MEME_RUBRIC = """
Binary label space:
- harmful: the target meme endorses, normalizes, mocks, threatens, dehumanizes, sexually objectifies, or humiliates a person or group; uses slurs or coded hate; promotes stereotypes or exclusion; celebrates violence, abuse, genocide, terrorism, self-harm, or exploitation; turns protected identity, tragedy, disability, body, gender, sexuality, religion, race, nationality, immigration status, or victimhood into the object of contempt.
- harmless: the target is benign, absurd, self-deprecating, neutral, descriptive, counterspeech, news-like, or uses edgy/dark wording without attacking or degrading a target. Mere mention of a protected group, tragedy, sex, politics, or violence is not harmful by itself.

Decision protocol:
1. Judge the target meme first: combine the image, overlaid/text content, implied speaker, target of the joke, and likely punchline.
2. Treat CAMR/retrieved samples as calibration, not ground truth. Use them when they are semantically close; discount them when they conflict with the target.
3. Sarcasm and irony often invert literal wording. Identify who is being mocked or harmed before assigning a label.
4. Label harmful when the meme punches down at a protected/vulnerable target, encourages harm, or makes abuse/hatred the joke.
5. Label harmless when the content is merely crude, ambiguous, factual, or references a sensitive topic without endorsement, ridicule, or attack.
6. If evidence is genuinely balanced, prefer the label supported by target-specific evidence over retrieved-label majority.
""".strip()


TOOL_DECISION_GUIDANCE = """
Use the rubric above. Return pred="harmful" only when this tool finds direct target-specific evidence of harm. Return pred="harmless" only when this tool finds evidence that the target is benign/counterspeech/non-attacking. Return pred="unknown" when the signal is weak, purely contextual, or outside this tool's scope.
Confidence calibration: use 0.80-1.00 only for clear target-specific evidence; 0.55-0.79 for plausible but incomplete evidence; 0.00-0.54 for weak, ambiguous, or out-of-scope signals. For pred="unknown", conf should usually be below 0.60.
Keep evidence short and concrete; do not reveal hidden chain-of-thought.
""".strip()


JSON_ONLY_RULE = "Return valid JSON only. Do not wrap it in markdown."
