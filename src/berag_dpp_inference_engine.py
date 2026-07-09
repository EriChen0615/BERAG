import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from vllm_api_backend import VLLMAPIBackend, VLLMGenerationResult


@dataclass
class BERAGDPPCandidate:
    subset_indices: Tuple[int, ...]
    text: str
    token_ids: List[int]
    token_logprobs: List[float]
    cumulative_logprob: float
    finish_reason: Optional[str] = None

    @property
    def avg_logprob(self) -> float:
        if not self.token_logprobs:
            return float("-inf")
        return sum(self.token_logprobs) / len(self.token_logprobs)


class BERAGDPPInferenceEngine:
    """
    BERAG-DPP inference engine backed by a long-lived vLLM HTTP server.

    Each round:
      1. estimate baseline entropy from no-evidence lookahead rollouts,
      2. estimate singleton conditional entropies from conditioned lookahead rollouts,
      3. construct a DPP over subsets from the resulting utilities,
      4. generate subset-conditioned candidates,
      5. keep the candidate with the highest joint score.
    """

    def __init__(
        self,
        backend: VLLMAPIBackend,
        debug: bool = False,
        segment_length: int = 8,
        num_look_ahead: int = 8,
        lookahead_rollout: int = 1,
        num_subset_samples: int = 4,
        max_subset_size: int = 2,
        beta: float = 1.0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: Optional[List[str]] = None,
        similarity_eps: float = 1e-6,
        max_dpp_rejection_rounds: int = 32,
        length_normalize_candidates: bool = False,
    ) -> None:
        self.backend = backend
        self.debug = debug
        self.segment_length = segment_length
        self.num_look_ahead = max(1, num_look_ahead)
        self.lookahead_rollout = max(1, lookahead_rollout)
        self.num_subset_samples = max(1, num_subset_samples)
        self.max_subset_size = max(1, max_subset_size)
        self.beta = beta
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stop = stop or []
        self.similarity_eps = similarity_eps
        self.max_dpp_rejection_rounds = max(1, max_dpp_rejection_rounds)
        self.length_normalize_candidates = length_normalize_candidates
        self._last_completed_candidates: List[Dict[str, Any]] = []

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[BERAG-DPP] {msg}", flush=True)

    @staticmethod
    def _extract_passage_text(passage: Union[str, Dict[str, Any]]) -> str:
        if isinstance(passage, str):
            return passage
        if isinstance(passage, dict):
            return str(passage.get("text", ""))
        return str(passage)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return " ".join((text or "").split())

    def _lcs_length(self, a: str, b: str) -> int:
        a_tokens = self._normalize_whitespace(a).split()
        b_tokens = self._normalize_whitespace(b).split()
        if not a_tokens or not b_tokens:
            return 0
        dp = [0] * (len(b_tokens) + 1)
        for tok_a in a_tokens:
            prev = 0
            for j, tok_b in enumerate(b_tokens, start=1):
                tmp = dp[j]
                if tok_a == tok_b:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = tmp
        return dp[-1]

    def _build_similarity_matrix(self, passages: Sequence[Union[str, Dict[str, Any]]]) -> np.ndarray:
        texts = [self._extract_passage_text(p) for p in passages]
        k = len(texts)
        mat = np.eye(k, dtype=np.float64)
        for i in range(k):
            tokens_i = max(1, len(self._normalize_whitespace(texts[i]).split()))
            for j in range(i + 1, k):
                tokens_j = max(1, len(self._normalize_whitespace(texts[j]).split()))
                lcs = self._lcs_length(texts[i], texts[j])
                denom = max(tokens_i, tokens_j)
                sim = float(lcs / denom) if denom > 0 else 0.0
                mat[i, j] = sim
                mat[j, i] = sim
        return self._project_to_psd(mat)

    def _project_to_psd(self, mat: np.ndarray) -> np.ndarray:
        sym = 0.5 * (mat + mat.T)
        vals, vecs = np.linalg.eigh(sym)
        vals = np.clip(vals, self.similarity_eps, None)
        psd = (vecs * vals) @ vecs.T
        diag = np.sqrt(np.clip(np.diag(psd), self.similarity_eps, None))
        psd = psd / np.outer(diag, diag)
        np.fill_diagonal(psd, 1.0)
        return psd

    def _format_evidence(self, subset_indices: Sequence[int], passages: Sequence[Union[str, Dict[str, Any]]]) -> str:
        return "\n\n".join(self._extract_passage_text(passages[i]) for i in subset_indices)

    def _build_prompt(
        self,
        input_context: Dict[str, Any],
        passages: Sequence[Union[str, Dict[str, Any]]],
        subset_indices: Sequence[int],
        history_text: str,
    ) -> str:
        instruction = str(input_context.get("text", ""))
        evidence = self._format_evidence(subset_indices, passages)
        if "<<<EVIDENCE>>>" in instruction:
            prompt = instruction.replace("<<<EVIDENCE>>>", evidence)
        else:
            prompt = f"{instruction}\n\n{evidence}"
        if history_text:
            prompt = f"{prompt}{history_text}"
        return prompt

    def _sample_rollouts(
        self,
        prompt: str,
        max_tokens: int,
        num_rollouts: int,
    ) -> List[VLLMGenerationResult]:
        requests_list = [
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "stop": self.stop,
                "logprobs": 1,
            }
            for _ in range(num_rollouts)
        ]
        results = self.backend.batch_generate(requests_list)
        for result in results:
            self.backend.require_logprob_capability(result)
        return results

    def _estimate_baseline_entropy(
        self,
        input_context: Dict[str, Any],
        passages: Sequence[Union[str, Dict[str, Any]]],
        history_text: str,
        lookahead_tokens: int,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        prompt = self._build_prompt(input_context, passages, [], history_text)
        results = self._sample_rollouts(prompt, lookahead_tokens, self.lookahead_rollout)
        entropies = [-float(result.cumulative_logprob) for result in results if result.num_generated_tokens > 0]
        baseline_entropy = float(sum(entropies) / len(entropies)) if entropies else float("inf")
        rollout_logs = []
        for idx, result in enumerate(results):
            rollout_logs.append(
                {
                    "rollout": idx,
                    "text": result.text,
                    "cumulative_logprob": float(result.cumulative_logprob),
                    "num_generated_tokens": int(result.num_generated_tokens),
                }
            )
            self._log(
                f"  baseline rollout={idx} generated_tokens={result.num_generated_tokens} "
                f"cum_logprob={result.cumulative_logprob:.4f} text={result.text[:160]!r}"
            )
        self._log(f"baseline entropy estimate={baseline_entropy:.4f} using lookahead_tokens={lookahead_tokens}")
        return baseline_entropy, rollout_logs

    def _estimate_singleton_utilities(
        self,
        input_context: Dict[str, Any],
        passages: Sequence[Union[str, Dict[str, Any]]],
        history_text: str,
        baseline_entropy: float,
        lookahead_tokens: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        utilities = np.zeros(len(passages), dtype=np.float64)
        qualities = np.zeros(len(passages), dtype=np.float64)
        singleton_logs: List[Dict[str, Any]] = []
        self._log(
            f"estimating singleton utilities: K={len(passages)} lookahead_tokens={lookahead_tokens} "
            f"lookahead_rollout={self.lookahead_rollout} history_chars={len(history_text)}"
        )
        for i in range(len(passages)):
            prompt = self._build_prompt(input_context, passages, [i], history_text)
            results = self._sample_rollouts(prompt, lookahead_tokens, self.lookahead_rollout)
            conditional_entropies = [-float(result.cumulative_logprob) for result in results if result.num_generated_tokens > 0]
            conditional_entropy = float(sum(conditional_entropies) / len(conditional_entropies)) if conditional_entropies else float("inf")
            utility = float(baseline_entropy - conditional_entropy)
            utilities[i] = utility
            qualities[i] = math.exp(self.beta * utility) if math.isfinite(utility) else 0.0
            evidence_preview = self._extract_passage_text(passages[i])[:120].replace("\n", " ")
            rollout_logs = [
                {
                    "rollout": idx,
                    "text": result.text,
                    "cumulative_logprob": float(result.cumulative_logprob),
                    "num_generated_tokens": int(result.num_generated_tokens),
                }
                for idx, result in enumerate(results)
            ]
            singleton_logs.append(
                {
                    "chunk_idx": int(i),
                    "conditional_entropy": float(conditional_entropy),
                    "information_gain": float(utility),
                    "quality": float(qualities[i]),
                    "rollouts": rollout_logs,
                }
            )
            self._log(
                f"  singleton chunk={i} conditional_entropy={conditional_entropy:.4f} "
                f"information_gain={utility:.4f} quality={qualities[i]:.6f} evidence_preview={evidence_preview!r}"
            )
        qualities = np.clip(qualities, self.similarity_eps, None)
        top_idx = np.argsort(-qualities)[: min(8, len(qualities))]
        self._log(
            "top singleton utilities: "
            + ", ".join(
                f"{int(idx)}=(u={float(utilities[idx]):.4f}, q={float(qualities[idx]):.6f})" for idx in top_idx
            )
        )
        return utilities, qualities, singleton_logs

    def _make_l_kernel(self, qualities: np.ndarray, similarity_matrix: np.ndarray) -> np.ndarray:
        sqrt_q = np.sqrt(np.clip(qualities, self.similarity_eps, None))
        l_kernel = similarity_matrix * np.outer(sqrt_q, sqrt_q)
        return self._project_to_psd(l_kernel)

    def _sample_dpp_once(self, l_kernel: np.ndarray) -> List[int]:
        vals, vecs = np.linalg.eigh(l_kernel)
        vals = np.clip(vals, 0.0, None)
        selected_cols = [i for i, lam in enumerate(vals) if np.random.rand() < (lam / (1.0 + lam))]
        if not selected_cols:
            return []
        v = vecs[:, selected_cols]
        items: List[int] = []
        while v.size > 0 and v.shape[1] > 0:
            probs = np.sum(v ** 2, axis=1)
            probs_sum = probs.sum()
            if probs_sum <= 0:
                break
            probs = probs / probs_sum
            item = int(np.random.choice(len(probs), p=probs))
            items.append(item)
            col_probs = v[item, :] ** 2
            col_sum = col_probs.sum()
            if col_sum <= 0:
                break
            col_probs = col_probs / col_sum
            j = int(np.random.choice(v.shape[1], p=col_probs))
            vj = v[:, j].copy()
            if abs(vj[item]) < self.similarity_eps:
                v = np.delete(v, j, axis=1)
            else:
                v = v - np.outer(vj, v[item, :] / vj[item])
                v = np.delete(v, j, axis=1)
            if v.size == 0 or v.shape[1] == 0:
                break
            v, _ = np.linalg.qr(v)
        return sorted(set(items))

    def _sample_bounded_subsets(self, l_kernel: np.ndarray, qualities: np.ndarray) -> List[Tuple[int, ...]]:
        subsets: List[Tuple[int, ...]] = []
        seen = set()
        quality_order = list(np.argsort(-qualities))
        fallback_singleton = (int(quality_order[0]),) if quality_order else (0,)

        attempts = 0
        max_attempts = self.num_subset_samples * self.max_dpp_rejection_rounds
        while len(subsets) < self.num_subset_samples and attempts < max_attempts:
            attempts += 1
            sampled = self._sample_dpp_once(l_kernel)
            if not sampled:
                sampled = list(fallback_singleton)
            if len(sampled) > self.max_subset_size:
                sampled = sorted(sampled, key=lambda idx: qualities[idx], reverse=True)[: self.max_subset_size]
            subset = tuple(sorted(sampled))
            if subset not in seen:
                seen.add(subset)
                subsets.append(subset)
                self._log(f"  accepted subset sample={subset}")

        while len(subsets) < self.num_subset_samples:
            idx = quality_order[len(subsets) % max(1, len(quality_order))] if quality_order else 0
            subset = (int(idx),)
            if subset not in seen:
                seen.add(subset)
                subsets.append(subset)
                self._log(f"  fallback subset sample={subset}")
            else:
                break

        self._log(f"sampled subsets: {[list(s) for s in subsets]}")
        return subsets or [fallback_singleton]

    def _subset_log_probability(self, subset: Sequence[int], l_kernel: np.ndarray) -> float:
        if not subset:
            return 0.0
        principal = l_kernel[np.ix_(subset, subset)]
        sign, logdet = np.linalg.slogdet(principal)
        if sign <= 0:
            return float("-inf")
        return float(logdet)

    def _generate_candidate_batch(
        self,
        input_context: Dict[str, Any],
        passages: Sequence[Union[str, Dict[str, Any]]],
        history_text: str,
        subsets: Sequence[Tuple[int, ...]],
        segment_length: int,
    ) -> List[BERAGDPPCandidate]:
        requests_list = []
        for subset in subsets:
            prompt = self._build_prompt(input_context, passages, subset, history_text)
            self._log(
                f"  generating candidate for subset={subset} prompt_chars={len(prompt)} "
                f"history_chars={len(history_text)} segment_length={segment_length}"
            )
            requests_list.append(
                {
                    "prompt": prompt,
                    "max_tokens": segment_length,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "stop": self.stop,
                    "logprobs": 1,
                }
            )
        results = self.backend.batch_generate(requests_list)
        candidates: List[BERAGDPPCandidate] = []
        for subset, result in zip(subsets, results):
            self.backend.require_logprob_capability(result)
            self._log(
                f"    candidate subset={subset} generated_tokens={result.num_generated_tokens} "
                f"cum_logprob={result.cumulative_logprob:.4f} finish_reason={result.finish_reason} "
                f"text={result.text[:160]!r}"
            )
            candidates.append(
                BERAGDPPCandidate(
                    subset_indices=tuple(subset),
                    text=result.text,
                    token_ids=result.token_ids,
                    token_logprobs=result.token_logprobs,
                    cumulative_logprob=result.cumulative_logprob,
                    finish_reason=result.finish_reason,
                )
            )
        return candidates

    def _score_candidate(self, candidate: BERAGDPPCandidate, subset_logprob: float) -> Tuple[float, float]:
        if candidate.token_ids and not candidate.token_logprobs and candidate.cumulative_logprob == 0.0:
            raise RuntimeError(
                "The vLLM server response did not include usable generated-token scoring metadata. "
                "BERAG-DPP needs either cumulative_logprob or generated-token logprobs for each completion."
            )
        generation_score = float(candidate.cumulative_logprob)
        if self.length_normalize_candidates and candidate.token_ids:
            generation_score = float(candidate.cumulative_logprob / len(candidate.token_ids))
        joint_score = generation_score + float(subset_logprob)
        return joint_score, generation_score

    def generate_berag_dpp(
        self,
        input_context: Dict[str, Any],
        passages: Sequence[Union[str, Dict[str, Any]]],
        max_new_tokens: int = 512,
        segment_length: Optional[int] = None,
        num_look_ahead: Optional[int] = None,
        lookahead_rollout: Optional[int] = None,
        num_subset_samples: Optional[int] = None,
        max_subset_size: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, Any]], np.ndarray]:
        if not passages:
            return "", [], np.zeros((0, 0), dtype=np.float64)

        m = segment_length if segment_length is not None else self.segment_length
        lookahead_tokens = num_look_ahead if num_look_ahead is not None else self.num_look_ahead
        lookahead_rollouts = lookahead_rollout if lookahead_rollout is not None else self.lookahead_rollout
        num_samples = num_subset_samples if num_subset_samples is not None else self.num_subset_samples
        subset_size = max_subset_size if max_subset_size is not None else self.max_subset_size
        old_num_samples = self.num_subset_samples
        old_subset_size = self.max_subset_size
        old_lookahead_rollout = self.lookahead_rollout
        self.num_subset_samples = max(1, num_samples)
        self.max_subset_size = max(1, subset_size)
        self.lookahead_rollout = max(1, lookahead_rollouts)
        try:
            similarity_matrix = self._build_similarity_matrix(passages)
            self._log(f"similarity matrix built once for K={len(passages)}")

            history_text = ""
            total_generated_tokens = 0
            all_round_logs: List[Dict[str, Any]] = []
            self._last_completed_candidates = []
            round_idx = 0
            terminated = False

            while total_generated_tokens < max_new_tokens and not terminated:
                remaining = max_new_tokens - total_generated_tokens
                round_m = min(m, remaining)
                self._log(
                    f"=== round={round_idx} total_generated_tokens={total_generated_tokens} "
                    f"remaining={remaining} segment_length={round_m} num_look_ahead={lookahead_tokens} "
                    f"lookahead_rollout={self.lookahead_rollout} ==="
                )

                baseline_entropy, baseline_rollouts = self._estimate_baseline_entropy(
                    input_context=input_context,
                    passages=passages,
                    history_text=history_text,
                    lookahead_tokens=lookahead_tokens,
                )
                singleton_utilities, qualities, singleton_logs = self._estimate_singleton_utilities(
                    input_context=input_context,
                    passages=passages,
                    history_text=history_text,
                    baseline_entropy=baseline_entropy,
                    lookahead_tokens=lookahead_tokens,
                )
                l_kernel = self._make_l_kernel(qualities, similarity_matrix)
                self._log(
                    "l-kernel diag preview: "
                    + ", ".join(f"{float(x):.4f}" for x in np.diag(l_kernel)[: min(8, len(l_kernel))])
                )
                subsets = self._sample_bounded_subsets(l_kernel, qualities)
                candidates = self._generate_candidate_batch(
                    input_context=input_context,
                    passages=passages,
                    history_text=history_text,
                    subsets=subsets,
                    segment_length=round_m,
                )
                if not candidates:
                    break

                scored: List[Tuple[float, float, float, BERAGDPPCandidate]] = []
                for candidate in candidates:
                    subset_logprob = self._subset_log_probability(candidate.subset_indices, l_kernel)
                    candidate_score, generation_score = self._score_candidate(candidate, subset_logprob)
                    scored.append((candidate_score, generation_score, subset_logprob, candidate))
                scored.sort(key=lambda item: item[0], reverse=True)

                for cand_score, generation_score, subset_logprob, cand in scored:
                    avg_logprob = cand.avg_logprob if cand.token_logprobs else None
                    self._log(
                        f"  scored candidate subset={cand.subset_indices} joint_score={cand_score:.4f} "
                        f"generation_score={generation_score:.4f} subset_score={subset_logprob:.4f} "
                        f"generation_logprob={cand.cumulative_logprob:.4f} avg_logprob={avg_logprob} "
                        f"text={cand.text[:160]!r}"
                    )

                best_score, best_generation_score, best_subset_logprob, best_candidate = scored[0]
                history_chars_before = len(history_text)
                history_text += best_candidate.text
                total_generated_tokens += len(best_candidate.token_ids)
                terminated = (
                    best_candidate.finish_reason in {"stop", "length", "eos_token"}
                    and len(best_candidate.token_ids) < round_m
                ) or len(best_candidate.token_ids) == 0

                top_quality_idx = np.argsort(-qualities)[: min(8, len(qualities))]
                round_log = {
                    "round": round_idx,
                    "remaining_budget": int(remaining),
                    "segment_length": int(round_m),
                    "num_look_ahead": int(lookahead_tokens),
                    "lookahead_rollout": int(self.lookahead_rollout),
                    "history_chars_before": history_chars_before,
                    "baseline_entropy": float(baseline_entropy),
                    "baseline_rollouts": baseline_rollouts,
                    "top_singletons": [
                        {
                            "chunk_idx": int(idx),
                            "information_gain": float(singleton_utilities[idx]),
                            "quality": float(qualities[idx]),
                        }
                        for idx in top_quality_idx
                    ],
                    "singleton_rollouts": singleton_logs,
                    "sampled_subsets": [list(s) for s in subsets],
                    "candidate_details": [
                        {
                            "subset_indices": list(c.subset_indices),
                            "score": float(score),
                            "generation_score": float(gen_score),
                            "subset_prior_score": float(subset_score),
                            "subset_logprob": float(c.cumulative_logprob),
                            "avg_logprob": float(c.avg_logprob) if c.token_logprobs else None,
                            "num_generated_tokens": len(c.token_ids),
                            "finish_reason": c.finish_reason,
                            "text": c.text,
                        }
                        for score, gen_score, subset_score, c in scored
                    ],
                    "winning_subset": list(best_candidate.subset_indices),
                    "winning_score": float(best_score),
                    "winning_generation_score": float(best_generation_score),
                    "winning_subset_prior_score": float(best_subset_logprob),
                    "winning_text": best_candidate.text,
                }
                all_round_logs.append(round_log)
                self._last_completed_candidates = [
                    {
                        "subset_indices": list(c.subset_indices),
                        "generated_text": c.text,
                        "score": float(score),
                        "generation_score": float(gen_score),
                        "subset_prior_score": float(subset_score),
                        "subset_logprob": float(c.cumulative_logprob),
                        "finish_reason": c.finish_reason,
                    }
                    for score, gen_score, subset_score, c in scored
                ]
                self._log(
                    f"round={round_idx} remaining={remaining} chose subset={best_candidate.subset_indices} "
                    f"tokens={len(best_candidate.token_ids)} joint_score={best_score:.4f} "
                    f"terminated={terminated} history_now={history_text[:200]!r}"
                )
                round_idx += 1

            return history_text, all_round_logs, similarity_matrix
        finally:
            self.num_subset_samples = old_num_samples
            self.max_subset_size = old_subset_size
            self.lookahead_rollout = old_lookahead_rollout
