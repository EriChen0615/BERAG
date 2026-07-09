import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests


@dataclass
class VLLMGenerationResult:
    text: str
    token_ids: List[int]
    token_logprobs: List[float]
    cumulative_logprob: float
    finish_reason: Optional[str] = None

    @property
    def num_generated_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def mean_logprob(self) -> float:
        if not self.token_logprobs:
            if self.num_generated_tokens == 0:
                return float("-inf")
            return self.cumulative_logprob / self.num_generated_tokens
        return sum(self.token_logprobs) / len(self.token_logprobs)


@dataclass
class VLLMScoreResult:
    text: str
    token_ids: List[int]
    token_logprobs: List[float]
    cumulative_logprob: float

    @property
    def num_scored_tokens(self) -> int:
        return len(self.token_ids)


class VLLMAPIBackend:
    """
    Thin HTTP client for a long-lived vLLM server.

    BERAG-DPP uses the server both for generation and for scoring a provided
    continuation under a prompt prefix. The scoring path relies on prompt-side
    logprobs from vLLM plus a local tokenizer offset map to slice out the
    continuation token region.
    """

    def __init__(
        self,
        server_host: str = "127.0.0.1",
        server_port: str = "5000",
        timeout: float = 300.0,
        max_workers: int = 16,
        tokenizer: Any = None,
    ) -> None:
        self.server_host = server_host
        self.server_port = server_port
        self.timeout = timeout
        self.max_workers = max_workers
        self.tokenizer = tokenizer

    @property
    def base_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    def _send_generate_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.put(
            url=f"{self.base_url}/generate",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _extract_token_logprobs(self, token_ids: Sequence[int], logprobs: Any) -> List[float]:
        if not isinstance(logprobs, list):
            return []
        token_logprobs: List[float] = []
        for pos, item in enumerate(logprobs):
            if item is None:
                continue
            if isinstance(item, (float, int)):
                token_logprobs.append(float(item))
                continue
            if isinstance(item, dict):
                target_token_id = token_ids[pos] if pos < len(token_ids) else None
                if "logprob" in item:
                    token_logprobs.append(float(item["logprob"]))
                    continue
                if "token_logprob" in item:
                    token_logprobs.append(float(item["token_logprob"]))
                    continue
                if target_token_id is not None:
                    target_key = str(target_token_id)
                    selected = item.get(target_key)
                    if isinstance(selected, dict) and "logprob" in selected:
                        token_logprobs.append(float(selected["logprob"]))
                        continue
                    if hasattr(selected, "logprob"):
                        token_logprobs.append(float(selected.logprob))
                        continue
                for value in item.values():
                    if isinstance(value, dict) and "logprob" in value:
                        token_logprobs.append(float(value["logprob"]))
                        break
        return token_logprobs

    def _coerce_result(self, output: Dict[str, Any]) -> VLLMGenerationResult:
        token_ids = [int(t) for t in (output.get("token_ids") or [])]
        token_logprobs = self._extract_token_logprobs(token_ids, output.get("logprobs"))
        cumulative_logprob = output.get("cumulative_logprob")
        if cumulative_logprob is None:
            cumulative_logprob = float(sum(token_logprobs)) if token_logprobs else 0.0
        return VLLMGenerationResult(
            text=output.get("text", ""),
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            cumulative_logprob=float(cumulative_logprob),
            finish_reason=output.get("finish_reason"),
        )

    def _extract_prompt_token_window(self, prompt: str, continuation: str) -> Sequence[int]:
        if self.tokenizer is None:
            raise RuntimeError(
                "VLLMAPIBackend.score_continuation requires a tokenizer so it can align prompt-side "
                "logprobs with the continuation span."
            )
        full_text = prompt + continuation
        tokenized = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = tokenized.get("offset_mapping") or []
        prompt_chars = len(prompt)
        continuation_positions: List[int] = []
        for idx, (start, end) in enumerate(offsets):
            if end <= prompt_chars:
                continue
            if start < prompt_chars < end:
                continuation_positions.append(idx)
            elif start >= prompt_chars:
                continuation_positions.append(idx)
        return continuation_positions

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        logprobs: int = 1,
        prompt_logprobs: Optional[int] = None,
    ) -> VLLMGenerationResult:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop or [],
            "logprobs": logprobs,
        }
        if prompt_logprobs is not None:
            payload["prompt_logprobs"] = prompt_logprobs
        raw = self._send_generate_request(payload)

        outputs = raw.get("outputs")
        if isinstance(outputs, list) and outputs:
            return self._coerce_result(outputs[0])

        text_list = raw.get("text")
        if isinstance(text_list, list) and text_list:
            return VLLMGenerationResult(
                text=str(text_list[0]),
                token_ids=[],
                token_logprobs=[],
                cumulative_logprob=0.0,
                finish_reason=None,
            )
        raise ValueError(f"Unexpected vLLM response format: {raw}")

    def batch_generate(self, requests_list: Iterable[Dict[str, Any]]) -> List[VLLMGenerationResult]:
        reqs = list(requests_list)
        if not reqs:
            return []

        def _run_one(kwargs: Dict[str, Any]) -> VLLMGenerationResult:
            return self.generate(**kwargs)

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(reqs))) as executor:
            return list(executor.map(_run_one, reqs))

    def score_continuation(self, prompt: str, continuation: str) -> VLLMScoreResult:
        if not continuation:
            return VLLMScoreResult(text="", token_ids=[], token_logprobs=[], cumulative_logprob=0.0)

        continuation_positions = list(self._extract_prompt_token_window(prompt, continuation))
        full_prompt = prompt + continuation
        raw = self._send_generate_request(
            {
                "prompt": full_prompt,
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "stop": [],
                "logprobs": 0,
                "prompt_logprobs": 1,
            }
        )
        prompt_token_ids = [int(t) for t in (raw.get("prompt_token_ids") or [])]
        prompt_logprobs = raw.get("prompt_logprobs") or []
        if not prompt_token_ids or not prompt_logprobs:
            raise RuntimeError(
                "The vLLM server response did not include prompt-side logprob metadata. "
                "BERAG-DPP information-gain scoring needs prompt_token_ids and prompt_logprobs."
            )

        if len(prompt_logprobs) != len(prompt_token_ids):
            usable = min(len(prompt_logprobs), len(prompt_token_ids))
            prompt_logprobs = prompt_logprobs[:usable]
            prompt_token_ids = prompt_token_ids[:usable]

        token_ids: List[int] = []
        token_logprobs: List[float] = []
        for pos in continuation_positions:
            if pos >= len(prompt_token_ids) or pos >= len(prompt_logprobs):
                continue
            token_id = prompt_token_ids[pos]
            extracted = self._extract_token_logprobs([token_id], [prompt_logprobs[pos]])
            if not extracted:
                continue
            token_ids.append(token_id)
            token_logprobs.append(float(extracted[0]))

        if not token_ids:
            raise RuntimeError(
                "Failed to align continuation tokens with prompt-side logprobs from the vLLM server."
            )

        return VLLMScoreResult(
            text=continuation,
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            cumulative_logprob=float(sum(token_logprobs)),
        )

    @staticmethod
    def require_logprob_capability(result: VLLMGenerationResult) -> None:
        if result.num_generated_tokens > 0 and not result.token_logprobs and result.cumulative_logprob == 0.0:
            raise RuntimeError(
                "The vLLM server response did not include usable generated-token scoring metadata. "
                "BERAG-DPP needs either cumulative_logprob or generated-token logprobs for each completion."
            )
