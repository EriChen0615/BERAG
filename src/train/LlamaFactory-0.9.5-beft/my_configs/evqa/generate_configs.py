#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from textwrap import dedent


CONFIG_ROOT = Path(__file__).resolve().parent
TOKENIZED_ROOT = Path("/workspace/projects/BERAG/outputs/tokenized")

MODEL_SPECS = {
    "2B": {
        "model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
        "output_slug": "qwen3-vl-2b",
    },
    "4B": {
        "model_name_or_path": "Qwen/Qwen3-VL-4B-Instruct",
        "output_slug": "qwen3-vl-4b",
    },
    "8B": {
        "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
        "output_slug": "qwen3-vl-8b",
    },
}

COMMON_MODEL_ARGS = {
    "image_max_pixels": 262144,
    "video_max_pixels": 16384,
    "trust_remote_code": True,
}

COMMON_LORA_ARGS = {
    "finetuning_type": "lora",
    "lora_target": "all",
    "lora_rank": 64,
    "lora_alpha": 128,
}

COMMON_DATA_ARGS = {
    "template": "qwen3_vl_nothink",
    "overwrite_cache": True,
    "media_dir": "/root/",
    "preprocessing_num_workers": 1,
    "preprocessing_batch_size": 8,
    "dataloader_num_workers": 4,
    "packing": False,
    "neat_packing": False,
}

COMMON_OUTPUT_ARGS = {
    "logging_steps": 10,
    "save_steps": 1000,
    "plot_loss": True,
    "overwrite_output_dir": False,
    "save_only_model": False,
    "report_to": "wandb",
}

COMMON_TRAIN_ARGS = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": "1.0e-5",
    "num_train_epochs": 1.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "bf16": True,
    "ddp_timeout": 180000000,
    "resume_from_checkpoint": None,
}

COMMON_EVAL_ARGS = {
    "val_size": 64,
    "per_device_eval_batch_size": 1,
    "eval_strategy": "steps",
    "eval_steps": 1000,
}

BEFT_PRIOR_ARGS = {
    "beft_hidden_state_offset": 4,
    "beft_prior_loss_factor":0.0,
    "beft_prior_head_lr": "1.0e-6",
    "beft_prior_modeling": "mlp_head",
    "beft_prior_head_num_layers": 2,
    "beft_prior_head_proj_dim": 1024,
    "beft_use_prior_head_loss": False,
}

TASK_SPECS = {
    "sft": {
        "file_name": "rag5_answer_sft.yaml",
        "dataset": "evqa_rag5_answer_sft",
        "cutoff_len": 4096,
        "tokenized_path": TOKENIZED_ROOT / "evqa_sft_qwen_sample64000_cutoff4096",
        "output_name": "rag5-answer-sft-r64-bs8-sample64000-max4096",
    },
    "beft": {
        "file_name": "beft_k2_prior_mlp.yaml",
        "dataset": "evqa_beft_k2_prior",
        "cutoff_len": 4096,
        "tokenized_path": TOKENIZED_ROOT / "evqa_beft_qwen_sample64000_cutoff4096",
        "output_name": "beft-k2-prior-mlp-r64-bs8-sample64000-max2500",
        "extra_model_args": {"flash_attn": "fa2"},
        "extra_method_args": BEFT_PRIOR_ARGS,
    },
}


def format_scalar(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)


def render_section(title: str, values: dict[str, object]) -> str:
    lines = [f"### {title}"]
    lines.extend(f"{key}: {format_scalar(value)}" for key, value in values.items())
    return "\n".join(lines)


def render_config(size: str, task: str) -> str:
    model_spec = MODEL_SPECS[size]
    task_spec = TASK_SPECS[task]
    output_slug = model_spec["output_slug"]

    model_args = {
        "model_name_or_path": model_spec["model_name_or_path"],
        **COMMON_MODEL_ARGS,
        **task_spec.get("extra_model_args", {}),
    }
    method_args = {
        "stage": task,
        "do_train": True,
        **COMMON_LORA_ARGS,
        **task_spec.get("extra_method_args", {}),
    }
    data_args = {
        "dataset": task_spec["dataset"],
        "template": COMMON_DATA_ARGS["template"],
        "cutoff_len": task_spec["cutoff_len"],
        **{k: v for k, v in COMMON_DATA_ARGS.items() if k != "template"},
        "tokenized_path": task_spec["tokenized_path"],
    }
    output_args = {
        "output_dir": f"saves/{output_slug}/lora/evqa/{task}/{task_spec['output_name']}",
        **COMMON_OUTPUT_ARGS,
    }

    blocks = [
        render_section("model", model_args),
        render_section("method", method_args),
        render_section("dataset", data_args),
        render_section("output", output_args),
        render_section("train", COMMON_TRAIN_ARGS),
        render_section("eval", COMMON_EVAL_ARGS),
    ]
    return "\n\n".join(blocks) + "\n"


def write_configs() -> None:
    for size in MODEL_SPECS:
        for task, task_spec in TASK_SPECS.items():
            path = CONFIG_ROOT / f"qwen3-vl-{size}" / task / task_spec["file_name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_config(size, task), encoding="utf-8")
            print(path.relative_to(CONFIG_ROOT))


def main() -> None:
    write_configs()


if __name__ == "__main__":
    main()
