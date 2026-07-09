"""
Simple NIAH (Needle-in-a-Haystack) adapter for BERAG-SG inference engine.

Provides SimpleNiahBERAGSGProvider (ModelProvider for the NIAH tester) that uses
the BERAG-SG segment-level beam search with chunk posterior and composite states.
Reuses chunking and prompt parsing from the ESF provider.
"""

import asyncio
from typing import Any, List, Optional, Union

from src.simple_niah_esf_provider import (
    _chunk_context_to_passages,
    _parse_prompt_to_context_and_question,
)


class SimpleNiahBERAGSGProvider:
    """
    ModelProvider implementation for the NIAH tester that uses the BERAG
    inference engine's BERAG-SG mode (segment-wise beam search, marginalized
    chunk scoring, optional composite-state rounds).
    """

    def __init__(
        self,
        engine: Any,
        backend: Any,
        model_name: str = "simple_niah_berag_sg",
        chunk_size: int = 512,
        num_of_chunks: Optional[int] = None,
        max_new_tokens: int = 300,
        needle: Optional[str] = None,
        inference_timeout: float = 3600.0,
        needles: Optional[List[str]] = None,
        segment_length: int = 4,
        berag_sg_beam_size: int = 2,
        max_composite_size: int = 1,
        log_dir: Optional[str] = None,
        berag_sg_top_p: Optional[float] = 0.9,
        berag_sg_temperature: float = 0.5,
        berag_sg_beam_prune: str = "diverse_beam_search",
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
        self.results_dir = None
        self.current_context_file_location = None
        self.segment_length = segment_length
        self.berag_sg_beam_size = berag_sg_beam_size
        self.max_composite_size = max_composite_size
        self.log_dir = log_dir
        self.berag_sg_top_p = berag_sg_top_p
        self.berag_sg_temperature = berag_sg_temperature
        self.berag_sg_beam_prune = berag_sg_beam_prune
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
        """Same structure as OpenAI/ESF provider so the tester does not change."""
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
        generated_tokens, *_ = await asyncio.wait_for(
            asyncio.to_thread(
                self.engine.generate_berag_sg,
                x,
                passages,
                max_new_tokens=self.max_new_tokens,
                results_dir=results_dir,
                results_stem=results_stem,
                segment_length=self.segment_length,
                berag_sg_beam_size=self.berag_sg_beam_size,
                max_composite_size=self.max_composite_size,
                log_dir=self.log_dir,
                berag_sg_top_p=self.berag_sg_top_p,
                berag_sg_temperature=self.berag_sg_temperature,
                berag_sg_beam_prune=self.berag_sg_beam_prune,
            ),
            timeout=self.inference_timeout,
        )
        self.last_completed_beams = getattr(self.engine, "_last_completed_beams", [])
        return self.decode_tokens(generated_tokens)
