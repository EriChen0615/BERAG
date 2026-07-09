# Design Document - Adding BEFT Trainer to LlamaFactory

# Introduction

LlamaFactory v0.9.5 supports tuning Qwen3-VL and other latest Vision-Language Models (VLMs). We'd like to add BEFT trainer so that we can fine-tune these VLMs on E-VQA.

We also want the addition to be clean and minimal, so that we can publish this codebase for training alongside our paper. 

# Background

## LlamaFactory Trainer Workflow (SFT Qwen3-VL)

On `llamafactory-cli train`, the code goes through:
* `cli.py`
* `launcher.py`
* `train/tuner.py`: here, the code goes to the "workflow" specified by `finetuning_args.stage in [pt, sft, rm, ppo, dpo, kto]`. 
    * We will illustrate the SFT workflow below, as it's cloest to our BEFT procedure. 
* `train/sft/workflow.py`: does the following thing:
    * load template, tokenizer, models
    * In `get_dataset()` call, the processed dataset is returned.
        * `get_dataset()` will select the dataset processing method based on the training stages. It will:
            * Load the actual data, depending on the type
            * Align the dataset with `converter.py`. I.e., training data in different formats (e.g., Alpaca, ShareGPT) get cast into a unified representation. 
            * It will then process the dataset with `_get_preprocessed_dataset()`
                * In this function, the `_get_dataset_processor()` function would select the data processor based on the finetuning stage.
                * These data processors are defined under `data/processor`. This is where tokenization happens.
    * define the data collator. `SFTDataCollatorWith4DAttentionMask`
        * The data collator uses 4D attention mask. Multiple examples can be packed into one sequence for efficient training. This is known as `packing` 
        * Note: in LlamaFactory `packing` may leaks between examples (i.e., the tokens in one example can attend to tokens in another example). `neat_packing` resolves this.
        * For our data collator, we will turn these packing techniques off. 
        * Our BEFT collator should inherit `MultiModalDataCollatorForSeq2Seq`, which is the parent class of `SFTDataCollatorWith4DAttentionMask` that supports multi-modal data collation. 
        * On `__call__()`, `MultiModalDataCollatorForSeq2Seq` does:
            * Expanded training examples (known as `features`) to batch constituents (e.g., batch_images, batch_videos, batch_input_ids). 
            * Multi-modal inputs are also processed and returned here via call to `self.template.mm_plugin.get_mm_inputs()`.
                * For Qwen-VL models, `pixel_values` and `image_grid_thw` are returned. 
                    * `pixel_values`: (num_patches, patch_dim)
                    * `image_grid_thw`: (num_images, 3), where the 3 are time, height, and width.
            * The `MultiModalDataCollatorForSeq2Seq` collator also handles MRoPE embeddings. It computes position ids for RoPE.
            * Finally, the `features` with `mm_inputs` updated are returned.
* Initialize the `CustomSeq2SeqTrainer`
    * The overriding `compute_loss` function returns the loss given the model and the inputs. 
        * We have simple `outputs = model(**inputs)` call here. 
    * This inherits from the transformers `Seq2SeqTrainer`. 
* Start training by calling `trainer.train()`
* Do Evaluation if `training_args.do_eval` is set.

## Bayesian Ensemble Fine-Tuning (BEFT)

BEFT is described in the [BERAG paper](https://arxiv.org/abs/2604.22678). To compute the next-token probability given a document set Z = [z1, z2, ..., zK], BERAG marginalize over all document singletons. 

P(y_t | x, y_1..t-1, Z) = \sum_{z in Z} P(y_t | x, y_1..t-1, z) P(z | x, y_1..t-1)

The first term is simply next-token likelihood given a specific passage. The second term is the document posterior which is computed via Baye's Rule:

P(z | x, y_1..t-1) = P(y_1..t-1 | x, z) P(z | x) / \sum_{u in Z} P(y_1..t-1 | x, u) P(u | x)

In training, we can obtain the next-token likelihood at ALL positions in the forward pass. The prior distribution, P(z | x), is obtained by passing the last token embedding (strictly speaking, the token at the -4 position for Qwen, the last token before the generation head) to a 2-layer MLP. 

## BEFT Training Logs

Apart from logging the overall BEFT loss, it is natural to log the *prior accuracy* when the ground-truth document labels are provided. 

Let k* be the number of ground-truth documents for a given instance. We set prior accuracy to be:

prior_acc  = Set(TopK(P(z|x))) == Set(The K GT docs)

In the single document case, this is equivalent to checking whether the Top-1 document is the gold document. 

Similarly, we can define *posterior accuracy*, where we compute accuracy at the last position of the output. This would show if the model's posterior accurately identifies the document. 

We can also log the entropy for the prior distribution, *prior entropy*, and the average entropy of the posterior distribution, *average posterior entropy*. The average is taken over the number of tokens in a given training instance.

# Implementation

In this section, we discuss the specific implementations required on LlamaFactory to perform BEFT.

## Overall Structure

We will define a new fine-tuning stage `beft`, and make sure that the associated hyper-parameters are defined.
    * This means a new folder `beft` under `src/train`, and new workflow files. 

We will add a new data processing file (a dataset processor). This processor will handle the data. 

We will provide a custom `compute_loss` function in our custom BEFT trainer. The trainer will inherit from `CustomSeq2SeqTrainer`. 

We will add the logging statements for BEFT metrics. 

## Data Processing

### BEFT Training Data Format

Below shows an example of BEFT training data (in the ShareGPT format):

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image> Answer the question given the evidence.\n\n<<<EVIDENCE>>>\n\nQuestion: What company made the laptop?"
    },
    {
      "from": "gpt",
      "value": "Apple"
    }
  ],
  "images": ["question_image.jpg"],
  "passages": [
    {
      "text": "<image>\nThis document shows a Dell laptop on a desk.",
      "images": ["passage_0.jpg"]
    },
    {
      "text": "<image>\nThe product label says Apple MacBook Pro.",
      "images": ["passage_1.jpg"]
    },
    {
      "text": "The warranty page discusses Lenovo accessories.",
      "images": []
    }
  ],
  "gt_passage_idx": [1]
}
```

As we can see, we specify the query image in the top-most `images` field. We provide the retrieved documents as a list of dictionary items. 
    Each dictionary contains two field: `text` and `images`. These represent the text and image of the passage. 
        Either field can be empty (i.e., pure-text or pure-image document is allowed)
    We use `<<<EVIDENCE>>>` to mark the position where the document singleton will be inserted.
    
The ground-truth (gold) passage is provided as a list. 

### Conversion 

We will need modify the `converter.py` file such that the ShareGPT conversion method preserves BEFT fields like `_passages` and  `_gt_passage_idx`. 

The converter converts training data in different format (Alpaca, ShareGPT, etc.) into a unified representation.

### BEFT Data Processor

We will define a BEFT data processor (`BeftDatasetProcessor`) to perform tokenization and prepare values for later collating. For each raw instance, the processor should produce K passage-conditioned sequences. 

This processor combines the query level media (`_images`) with the passsage-local media (`passage[k]['images']`)

It then proceeds similarly like the data processor for SFT.

The dataset processor should output: `all_input_ids`, `all_attention_mask`, `all_labels`, `all_passage_images`, `gt_passage_idx`. 


### BEFT Data Collator

We will need to add a `BeftDataCollator` which inherits from `MultiModalDataCollatorForSeq2Seq`. The data collator does the following:
    * Given one BEFT training example, the collator expands its K passage items into K ordinary multi-modal features. It will also mark the feature that corresponds to gold document(s) with the field `is_gt_passage`. 
    * `packing` and `neat_packing` are explicitly disabled. 
    * Initially, we would require `per_device_train_batch_size=1`. Internally, the model would handle K (N.B., we will use a small K in training) sequences per forward.
    * The data collator would similarly call `get_mm_inputs` (by delegating to `MultiModalDataCollatorForSeq2Seq`) to obtain multi-modal features for forward.






## BEFT Training

We will implement `CustomSeq2SeqBEFTTrainer` that inherits from `CustomSeq2SeqTrainer`. All BEFT code should live under `train/beft/`. 

In the `compute_loss` function, we can expect the input to be the K passage conditioned sequences from ONE raw training instance. 

We will run forward pass with `output_hidden_states=True`, so that we can feed the returned hidden state to the prior MLP head to compute the prior P(z|x). 

Here, we show how to compute the BEFT loss efficiently with torch. 

```python
def compute_beft_loss(
    token_logps: torch.Tensor,
    prior_logits: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Compute the BEFT marginalized next-token loss.

    Args:
        token_logps:
            Float tensor of shape [K, T].
            token_logps[k, i] is log p_theta(y_i | x, z_k, y_<i), i.e. the
            log-probability of the i-th answer token when conditioning on the
            k-th passage.

        prior_logits:
            Optional float tensor of shape [K] or [K, 1].
            These are unnormalized logits for p_phi(z_k | x). If omitted, BEFT
            uses a uniform passage prior.

    Returns:
        loss:
            Scalar tensor: -sum_i log p(y_i | x, Z, y_<i).

        posterior_logprobs:
            Float tensor of shape [K, T].
            posterior_logprobs[k, i] is log p(z_k | x, Z, y_<i), the passage
            posterior before observing token y_i.

        prior_logprobs:
            Float tensor of shape [K].
            log p_phi(z_k | x), or uniform log-prior if prior_logits is None.
    """
    if token_logps.ndim != 2:
        raise ValueError(f"token_logps must have shape [K, T], got {tuple(token_logps.shape)}.")

    num_passages = token_logps.size(0)

    if prior_logits is None:
        prior_logprobs = token_logps.new_zeros(num_passages)
    else:
        prior_logprobs = torch.log_softmax(prior_logits.view(num_passages), dim=0)

    prefix_logps = torch.cumsum(token_logps, dim=1) - token_logps
    posterior_scores = prefix_logps + prior_logprobs[:, None]
    posterior_logprobs = posterior_scores - torch.logsumexp(posterior_scores, dim=0, keepdim=True)

    token_marginal_logprobs = torch.logsumexp(token_logps + posterior_logprobs, dim=0)
    loss = -token_marginal_logprobs.sum()

    return loss, posterior_logprobs, prior_logprobs
```

In the function above, we have assumed we have `token_logps`, which corresponds to the labels of the K sequences. We will define a helper function below to obtain `token_logps` for us:

```python
from llamafactory.extras.constants import IGNORE_INDEX


def get_answer_token_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_pad_token_id: int = IGNORE_INDEX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Extract answer-token log-probabilities from causal LM logits.

    Args:
        logits: Float tensor of shape [K, L, V].
            Model logits for K passage-conditioned rows.

        labels: Long tensor of shape [K, L].
            Labels use `label_pad_token_id` for non-answer tokens. The valid
            label tokens should be the same answer across all K passages.

    Returns:
        token_logps: Float tensor of shape [K, T].
            token_logps[k, i] = log p_theta(y_i | x, z_k, y_<i).

        answer_lengths: Long tensor of shape [K].
            Number of supervised answer tokens per passage row.
    """
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} is incompatible with labels shape {tuple(labels.shape)}.")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].clone()
    loss_mask = shifted_labels != label_pad_token_id

    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    all_token_logps = torch.gather(
        shifted_logits.log_softmax(dim=-1),
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    answer_lengths = loss_mask.sum(dim=-1)
    if not torch.all(answer_lengths == answer_lengths[0]):
        raise ValueError(f"BEFT expects equal answer lengths across passages, got {answer_lengths.tolist()}.")

    token_logps = torch.stack(
        [row_logps[row_mask] for row_logps, row_mask in zip(all_token_logps, loss_mask)],
        dim=0,
    )
    return token_logps, answer_lengths
```

The full `compute_loss` function will do
* pop BEFT-only fields like `is_gt_passage`
* call `model(**inputs)`, returning output hidden states if using prior head
* compute `token_logps` with the helper function
* compute BEFT loss with `compute_beft_loss`. 
* (optinally) add multi-hot BCE prior supervision from `is_gt_passage`. 

## BEFT Logging

Append detached scaler metrics. 

Log : beft_loss, prior_loss, total_loss, prior_acc, posterior_acc_last, posterior_acc_mean, prior_entropy, posterior_entropy_mean, and num_gt_docs. 

