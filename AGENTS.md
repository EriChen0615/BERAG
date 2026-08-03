## Communication Style

Respond clearly and directly in no more than 2 paragraphs, unless asked otherwise. Break down big chunk of information and communicate piece-by-piece with explicit confirmation from the user, like an efficient Engineer.

When asked multiple questions, answer each question in no more than 2 paragraphs, by first repeating the question in the format `> <question>` and then give your answer.

After discussing experimental results, ask for user permission before adding those findings to repository documents such as files under `docs/`.

Before running analysis, evaluation, training, or inference commands, select the appropriate environment using the guidance below.

## Environments and Execution

- Run commands from the repository root, normally `/workspace/projects/BERAG`, and use repository-relative paths in scripts and patches.
- This repository is used on both Cambridge HPC and RunPod. Do not assume Slurm, environment modules, or Cambridge filesystem paths exist on RunPod.
- For LlamaFactory training and lightweight dataset checks on RunPod, activate `scripts/activate_train_env.sh` or use `src/train/LlamaFactory-0.9.5-beft/.venv/bin/python` explicitly.
- InfoSeek official evaluation has its own environment at `src/eval/.venv`. EVQA BEM evaluation uses `src/eval/bem` where requested.
- Inference launchers default to `PYTHON_BIN=python`; run them from an already activated environment containing the local vLLM-BERAG build, or set `PYTHON_BIN` explicitly.
- `old_scripts/hpc_activate_env_py310_infer.sh` is HPC-specific and contains module/Conda paths that are not valid on RunPod. The former `scripts/hpc_activate_env_py310_infer.sh` path is stale on this checkout.
- Do not install or upgrade packages merely to run a check. First locate an existing suitable environment; report when a test dependency such as pytest is unavailable.

## Editing Files on This Machine

- The worktree is often dirty. Treat all existing modifications and untracked files as user work. Inspect `git status --short` and a path-scoped `git diff` before editing; never reset, restore, or reformat unrelated files.
- Prefer small unified patches from the repository root. The normal editor is `apply_patch`.
- In the current Codex/VS Code sandbox, the built-in `apply_patch` helper may panic with `permission profiles requiring direct runtime enforcement are incompatible with --use-legacy-landlock`. If that happens, use the system `patch -p1` command with a unified diff as the narrow fallback. Keep paths relative and patch only intended files.
- The system patch utility may create `.orig` or `.rej` files. Inspect them, then remove only artifacts created by the current operation. Always run a scoped `git status` afterward.
- Do not replace whole files when a localized patch suffices. Preserve comments, launcher defaults, and unrelated user changes.
- Standard final checks are `git diff --check -- <paths>`, `python -m py_compile <python files>`, and `bash -n <shell files>`. Run focused CPU assertions or unit tests in proportion to the change.

## Inference Architecture

- `src/infer/evqa_vllm_berag_inference.py` is the shared RAG/BERAG inference implementation. Keep dataset-specific behavior out of it except for small reusable hooks.
- `src/infer/infoseek_vllm_berag_inference.py` is the InfoSeek adapter. The canonical InfoSeek launcher is `scripts/infer/infoseek/run_sft_beft.sh`; model-specific wrappers delegate to it.
- The modified engine source is a separate checkout at `/workspace/projects/vllm-berag`. Apply scheduler, model-loader, or BERAG state-transition changes there, not under the BERAG repository's vendored-looking paths.
- Unless explicitly requested, output JSONL must not contain full prompts or document text. Save passage IDs and compact telemetry; `--save-full-prompts` is opt-in.
- The user runs model smoke tests. Codex should perform syntax, shell, data-shape, and CPU-only logic checks but leave GPU inference smoke runs to the user.

## BERAG Invariants and Prior Offsets

- One entry in `PreparedExample.documents` is one BERAG latent branch. Any branch ID list and probability list must remain one-to-one and in the same order.
- Qwen2-VL, Qwen3-VL, and the current Gemma4 BEFT checkpoints use training `beft_hidden_state_offset: 4`, whose compatible prompt-relative inference position is `-5` because training selects `L - offset - 1`.
- When continuing from an incomplete assistant turn containing `[ANSWER]`, retain the assistant generation header/template semantics and resolve the extra token offset once per tokenizer, not once per example.
- BERAG prior/posterior telemetry contains aligned normalized log values in `berag_log_prior` and `berag_log_posterior`. Preserve `None` for unavailable or pruned branches.
- The pseudo retrieval-failure document is appended once to the ensemble, not once per retrieved page or section, and its branch metadata must remain aligned.

## InfoSeek Data and Section Explosion

- The processed full M2KR InfoSeek dataset is normally `outputs/jinghong_chen/Infoseek-testfull-with-retrieval`. Retrieval entries are page-level Wikipedia URLs.
- The EchoSight KB is normally `/workspace/projects/Infoseek-retrieval/data/echosight/infoseek_100k_wiki/wiki_100_dict_v4.json`. It stores aligned `section_titles` and `section_texts` for each page.
- `--explode-page-sections` first selects the requested top-K pages and then expands nonempty sections into separate evidence branches. `--max-exploded-docs-per-page` caps each page deterministically in source order and defaults to 5.
- Every exploded section uses the same evidence syntax as a page: `Title: {page title} — {section title}\tContent: {section text}\n`. Section titles and texts are indexed together and their lengths are validated.
- Section IDs are deterministic: `{page_url}::section::{section_index}`. Preserve both unique section IDs and aligned parent page URLs.
- Three entries in the 100K KB have no nonempty section text. They receive one title-only fallback branch, matching the information available from the prior page reconstruction.
- With the default per-page cap of 5, explosion creates at most `5 * K` section branches. InfoSeek inference warns when `K * max_exploded_docs_per_page` is not smaller than `num_accumulator_rows`; BERAG also needs one mixture row.
- Inference output records `retrieved_page_ids`, `retrieved_section_ids`, `berag_branch_passage_ids`, and aligned `berag_branch_parent_page_ids`. With `EXPLODE_PAGE_SECTIONS=1`, the launcher adds `-explode-sections` to the experiment directory so resumed outputs cannot mix modes.

## InfoSeek Evaluation

- `src/eval/evaluate_infoseek_predictions.py` runs the official answer evaluator and reports retrieval, prior, and posterior Recall@K.
- Retrieval Recall@K remains page-level and uses the original `retrieved_page_ids` ranking.
- For exploded BERAG outputs, evaluation groups section branches by parent page and computes each page score by `logsumexp` over section log-probabilities, equivalent to summing section probabilities. It then ranks pages for prior/posterior Recall@K.
- Do not obtain page rankings by merely deduplicating the sorted section list; that is not equivalent to summing probability mass.
- Preserve section-level telemetry for diagnosis while reporting page-level Recall@K for comparability with InfoSeek retrieval.
- The official InfoSeek aggregate score and `mean_example_score` have different subset semantics. When evaluating subsets, use `mean_example_score` for the direct mean over supplied examples and discuss official aggregate limitations explicitly.
