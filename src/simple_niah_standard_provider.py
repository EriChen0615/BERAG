"""
Simple NIAH (Needle-in-a-Haystack) provider using standard autoregressive generation.

Provides SimpleNiahStandardProvider (ModelProvider for the NIAH tester) with the same
prompt interface as the ESF provider but a single full-context forward + model.generate().
"""

import asyncio
from typing import Any, List, Optional, Union

from src.simple_niah_esf_provider import _parse_prompt_to_context_and_question


class SimpleNiahStandardProvider:
    """
    ModelProvider implementation for the NIAH tester that uses standard text generation:
    full prompt (context as single passage) then model.generate(). No BERAG engine.
    """

    def __init__(
        self,
        backend: Any,
        model_name: str = "simple_niah_standard",
        max_new_tokens: int = 300,
        needle: Optional[str] = None,
    ):
        self.backend = backend
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.needle = (needle or "").strip() or None
        self._tokenizer = backend.tokenizer
        # Tester may set these; no beam log for standard gen, but attributes exist for compatibility
        self.results_dir = None
        self.current_context_file_location = None

    def encode_text_to_tokens(self, text: str) -> List[int]:
        enc = self._tokenizer.encode(text, add_special_tokens=False)
        return list(enc.ids) if hasattr(enc, "ids") else list(enc)

    def decode_tokens(
        self,
        tokens: Union[List[int], "tokenizers.Encoding"],
        context_length: Optional[int] = None,
    ) -> str:
        if hasattr(tokens, "ids"):
            tokens = list(tokens.ids)
        if context_length is not None:
            tokens = tokens[:context_length]
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def generate_prompt(
        self,
        context: str,
        retrieval_question: str,
    ) -> Union[str, List[dict]]:
        """Same structure as ESF provider so the tester does not change."""
        return [
            {
                "role": "system",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct",
            },
            {"role": "user", "content": context},
            {
                "role": "user",
                "content": f"{retrieval_question} Don't give information outside the document or repeat your findings",
            },
        ]

    async def evaluate_model(self, prompt: Union[str, List[dict]]) -> str:
        context, retrieval_question = _parse_prompt_to_context_and_question(prompt)
        template = (
            "Answer based only on the following evidence.\n\n<<<EVIDENCE>>>\n\nQuestion: "
            + (retrieval_question or "")
        )
        x = {"text": template, "image": None}
        # Full context as single "passage" (zk)
        zk = context or ""
        inputs = self.backend._get_prompt_only_inputs(x, zk)
        generated_ids = await asyncio.to_thread(
            self.backend.generate_standard, inputs, self.max_new_tokens
        )
        return self.decode_tokens(generated_ids)
