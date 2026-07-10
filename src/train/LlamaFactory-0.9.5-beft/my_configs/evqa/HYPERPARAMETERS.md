# EVQA Training Hyperparameters

These configs follow the old `LlamaFactory-0.9.3-beft/my_configs` BEFT/SFT style where the keys still exist in `LlamaFactory-0.9.5-beft`.

## Common LoRA/training/eval settings

- `finetuning_type: lora`
- `lora_target: all`
- `lora_rank: 64`
- `lora_alpha: 128`
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 8`
- `learning_rate: 1.0e-5`
- `num_train_epochs: 1.0`
- `lr_scheduler_type: cosine`
- `warmup_ratio: 0.1`
- `bf16: true`
- `val_size: 64`
- `per_device_eval_batch_size: 1`
- `eval_strategy: steps`
- `eval_steps: 20000`

## BEFT prior-head settings

`0.9.5-beft` renamed the old PPL/BEFT prior-head arguments. The active configs use the defined `beft_*` names:

- old `ppl_hidden_state_offset: 4` -> `beft_hidden_state_offset: 4`
- old `ppl_prior_loss_factor: 1.0` -> `beft_prior_loss_factor: 1.0`
- old `prior_head_lr: 1.0e-6` -> `beft_prior_head_lr: 1.0e-6`
- old `prior_modeling: mlp_head` / `ppl_prior_modeling: mlp_head` -> `beft_prior_modeling: mlp_head`
- old `prior_head_num_of_layers: 2` / `ppl_prior_head_num_of_layers: 2` -> `beft_prior_head_num_layers: 2`
- old `prior_head_proj_dim: 1024` / `ppl_prior_head_proj_dim: 1024` -> `beft_prior_head_proj_dim: 1024`
- `beft_use_prior_head_loss: true` enables prior-head BCE supervision.

Not defined in `0.9.5-beft`:

- `ppl_prior_loss_type`; the current BEFT trainer uses `BCEWithLogitsLoss`, equivalent to logistic binary cross entropy.
- `use_ppl_loss`; the current `stage: beft` trainer directly computes the BEFT marginalized loss and does not expose this old switch.
