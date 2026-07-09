"""
Simple NIAH (Needle-in-a-Haystack) adapter for BERAG ESF inference engine.

Provides SimpleNiahESFProvider (ModelProvider for the NIAH tester) and
SimpleNiahLocalEvaluator (local evaluator without API).
"""

import asyncio
import re
from typing import Any, List, Optional, Union


def _parse_prompt_to_context_and_question(prompt: Union[str, List[dict]]) -> tuple:
    """
    Extract context string and retrieval question from NIAH prompt.
    prompt is typically list of message dicts: [system, user: context, user: question].
    """
    if isinstance(prompt, str):
        return "", prompt
    if not isinstance(prompt, (list, tuple)) or len(prompt) < 3:
        return "", ""
    # Second message is user with long context, third is user with question
    context = ""
    question = ""
    for m in prompt:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", c) if isinstance(c, dict) else str(c) for c in content
            )
        if role == "user":
            if not context:
                context = content or ""
            else:
                question = content or ""
                break
    return context, question


def _chunk_context_to_passages(
    context: str,
    tokenizer: Any,
    chunk_size: int = 512,
    num_of_chunks: Optional[int] = None,
    overlap_words: int = 100,
) -> List[str]:
    """Split context into sliding-window passage strings using word overlap.

    This is intentionally tokenizer-free: we split on whitespace via `context.split()`.

    Args:
        context: Full context string.
        tokenizer: Kept for backward compatibility with existing callers; not used here.
        chunk_size: Passage size in words (when num_of_chunks is not set).
        num_of_chunks: If provided, derives `chunk_size` (in words) as ceil(len(words) / num_of_chunks).
        overlap_words: Number of words to overlap between consecutive passages.
    """
    if not context.strip():
        return [""]
    words = context.split()
    if not words:
        return [""]

    chunk_words = int(chunk_size) if chunk_size is not None else 1
    chunk_words = max(1, chunk_words)
    if num_of_chunks is not None and num_of_chunks > 0:
        chunk_words = max(1, (len(words) + num_of_chunks - 1) // num_of_chunks)

    overlap_words = int(overlap_words) if overlap_words is not None else 0
    overlap_words = max(0, overlap_words)
    # Ensure overlap_words is strictly less than chunk_words to keep step >= 1.
    if overlap_words >= chunk_words:
        overlap_words = chunk_words - 1

    step = max(1, chunk_words - overlap_words)
    passages = []
    for i in range(0, len(words), step):
        chunk_words_slice = words[i : i + chunk_words]
        if not chunk_words_slice:
            continue
        passages.append(" ".join(chunk_words_slice))

    return passages if passages else [""]


class SimpleNiahESFProvider:
    """
    ModelProvider implementation for the NIAH tester that uses BERAG ESF
    inference engine with a local HF backend.
    """

    def __init__(
        self,
        engine: Any,
        backend: Any,
        model_name: str = "simple_niah_berag_esf",
        chunk_size: int = 512,
        num_of_chunks: Optional[int] = None,
        max_new_tokens: int = 300,
        needle: Optional[str] = None,
        inference_timeout: float = 3600.0,
        needles: Optional[List[str]] = None,
    ):
        self.engine = engine
        self.backend = backend
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.num_of_chunks = num_of_chunks
        self.max_new_tokens = max_new_tokens
        self.needle = (needle or "").strip() or None
        self.inference_timeout = inference_timeout
        self.needles = needles if needles else None
        self._tokenizer = backend.tokenizer
        # So tester's hasattr checks pass; tester sets these before each evaluate_model
        self.results_dir = None
        self.current_context_file_location = None
        # Completed beams (scores + posteriors) from last generate; set after evaluate_model
        self.last_completed_beams = []

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
        """Same structure as OpenAI provider so the tester does not change."""
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
        passages = _chunk_context_to_passages(
            context,
            self._tokenizer,
            self.chunk_size,
            num_of_chunks=self.num_of_chunks,
        )
        template = (
            "Answer based only on the following evidence.\n\n<<<EVIDENCE>>>\n\nQuestion: "
            + (retrieval_question or "")
        )
        x = {"text": template, "image": None}
        if self.needle is not None:
            x["needle"] = self.needle
        if self.needles:
            gold_chunks_per_needle = []
            for n in self.needles:
                needle_stripped = (n or "").strip()
                k_found = -1
                for k, p in enumerate(passages):
                    if needle_stripped and needle_stripped in (p or ""):
                        k_found = k
                        break
                gold_chunks_per_needle.append(k_found)
            x["gold_chunks_per_needle"] = gold_chunks_per_needle
        results_dir = getattr(self, "results_dir", None)
        results_stem = getattr(self, "current_context_file_location", None)
        if results_dir is not None and results_stem is not None:
            x["results_dir"] = results_dir
            x["results_stem"] = results_stem
        generated_tokens, _, _ = await asyncio.wait_for(
            asyncio.to_thread(
                self.engine.generate,
                x,
                passages,
                self.max_new_tokens,
            ),
            timeout=self.inference_timeout,
        )
        self.last_completed_beams = getattr(self.engine, "_last_completed_beams", [])
        return self.decode_tokens(generated_tokens)


class SimpleNiahLocalEvaluator:
    """
    Local evaluator for simple_niah: scores response by relevance to the needle.
    Implements evaluate_response(response: str) -> int (1--10).
    """

    def __init__(
        self,
        needle: str,
        question_asked: Optional[str] = None,
    ):
        self.needle = (needle or "").strip()
        self.question_asked = question_asked or ""
        # Normalize for substring match: collapse whitespace and lower
        self._needle_norm = re.sub(r"\s+", " ", self.needle).lower() if self.needle else ""

    def evaluate_response(self, response: str) -> int:
        if not response or not self._needle_norm:
            return 1
        resp_norm = re.sub(r"\s+", " ", response).lower()
        if self._needle_norm in resp_norm:
            return 10
        # Partial: check word overlap
        needle_words = set(self._needle_norm.split())
        resp_words = set(resp_norm.split())
        overlap = len(needle_words & resp_words) / max(len(needle_words), 1)
        if overlap >= 0.8:
            return 8
        if overlap >= 0.5:
            return 5
        if overlap >= 0.2:
            return 3
        return 1


class SimpleNiahMultiNeedleEvaluator:
    """
    Local evaluator for multi-needle NIAH: scores response by how many needles are retrieved.
    Implements evaluate_response(response: str) -> int (1--10).
    """

    def __init__(
        self,
        needles: List[str],
        question_asked: Optional[str] = None,
    ):
        self.needles = [n for n in needles if (n or "").strip()]
        self.question_asked = question_asked or ""
        self._needles_norm = [
            re.sub(r"\s+", " ", n).strip().lower() for n in self.needles if n
        ]
        self._needles_norm = [n for n in self._needles_norm if n]

    def evaluate_response(self, response: str) -> int:
        if not response or not self._needles_norm:
            return 1
        resp_norm = re.sub(r"\s+", " ", response).lower()
        found = sum(1 for n in self._needles_norm if n in resp_norm)
        n_total = len(self._needles_norm)
        if n_total == 0:
            return 1
        if found == n_total:
            return 10
        if found >= 1:
            return int(1 + round(8 * found / n_total))
        return 1
