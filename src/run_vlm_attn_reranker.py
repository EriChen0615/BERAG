"""
This script runs the VLM attention-based reranker on a given (retrieval-augmented) dataset (e.g., EVQA). 

INPUTS:
- base_inference_filepath: path to the annotated inference file of the base model
- dataset_path: path to the retrieval-augmented dataset
- dataset_name: used to obtain passage set
- model_path: path to the VLM model
- attn_rank_mode: modes of attention-based ranking score computation. E.g., 'sum', 'late-interaction', etc.
- retrieval_field: either 'retrieved_passage' or 'reranked_passage'
- retrieval_topk: must be the same as that used in `base_inference_filepath`
- batch_size: batch_size used to obtain attention scores (i.e., forward)
- output_dir: directory to output files/graphs

OUTPUTS:
- retriever_recall@K: recall value using the retriever only
- vlm_rerank_recall@K: recall value after vlm attn-based reranking
- acc|attn-topK: accuracy given that the Ground-Truth (GT) document is ranked as the TopK document
- bin_digram(accuracy versus GT doc probability): a bin diagram with x-axis = probability assigned to the GT document; y-axis = accuracy
- inference_file_with_attn_scores: base inference file with additional columns ['passage_probs', 'attn_mode']

ALGORITHM:
1. Read & Format Dataset.
    * Form RAG-TopK prompt
    * Annotate evidence_spans, attn_source_span, and attn_cal_span for each instance. 
2. Run model forward, get attention scores and compute reranking scores
    * Ranking score computation should be based on `attn_rank_mode`
    * Compute passage attentions (probability) for each instance.
3. Compute metrics
    * recall@K under the original retriever and the vlm attention
    * accuracy (with generations taken from `base_inference_filepath`) given TopK attention.
    * bin diagram (acc versus probability)
"""

import sys
sys.path.append('./src')
sys.path.append('./third_party/LLaMAFactory/src/llamafactory/train/attn_sft')
from attn_loss import _compute_attn_reranking_scores
from vqa_datasets import load_passages
from curate.ragk_answer_attn_sft import (
    VLM_PROMPT_FOR_VQA,
    EVIDENCE_START_TOKEN,
    EVIDENCE_END_TOKEN,
    ATTN_SOURCE_START_TOKEN,
    ATTN_SOURCE_END_TOKEN,
    ATTN_CALI_START_TOKEN,
    ATTN_CALI_END_TOKEN
)
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from torch.utils.data import DataLoader
from functools import partial
import torch
import argparse
from qwen_vl_utils import process_vision_info
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

def add_prefix_and_form_prompt(batch, args, pid_to_content_map):
    """Process a batch of data efficiently using HuggingFace batched processing"""
    batch_size = len(batch['img_path'])
    
    # Process image paths
    img_paths = [f"{args.img_basedir}/{p}" for p in batch['img_path']]
    
    # Pre-allocate prompts list for better performance
    prompts = [''] * batch_size
    gt_evidence_labels = [[0] * args.topk_docs for _ in range(batch_size)]
    
    # Process each item in the batch
    for idx in range(batch_size):
        # Build evidence parts efficiently
        evidence_parts = []
        for i in range(args.topk_docs):
            passage_dict = batch[args.retrieval_field][idx][i]
            if passage_dict['passage_id'] in batch['pos_item_ids'][idx]:
                gt_evidence_labels[idx][i] = 1
            if 'text' in passage_dict:
                text = passage_dict['text']
            else:
                text = pid_to_content_map[passage_dict['passage_id']]
            text = ' '.join(text.split(' ')[:512]) #NOTE restrict length
            
            evidence_part = (
                "[EVIDENCE]"
                f"{EVIDENCE_START_TOKEN}"
                f"Title: {passage_dict['passage_id']}\t"
                f"Content: {text}"
                f"{EVIDENCE_END_TOKEN}\n"
            )
            evidence_parts.append(evidence_part)
        
        question_part = None
        if args.attn_calibration_span == 'question_token':
            question_part = f"\n{ATTN_CALI_START_TOKEN}[QUESTION]{ATTN_CALI_END_TOKEN}"
        else:
            question_part = f"\n[QUESTION] "
        if args.attn_source_span == 'question':
            question_part += f"{ATTN_SOURCE_START_TOKEN}{batch['question'][idx]}{ATTN_SOURCE_END_TOKEN}"
        else:
            question_part += f"{batch['question'][idx]}"

        prompt = VLM_PROMPT_FOR_VQA \
            + ''.join(evidence_parts) \
            + question_part

        prompts[idx] = prompt
    
    # Update batch with processed data
    batch['img_path'] = img_paths
    batch['prompt'] = prompts
    batch['gt_evidence_labels'] = gt_evidence_labels
    return batch

def tokenize_and_get_spans(batch, processor, space_token_id, device):
    # modify tokenizer
    tokenizer = processor.tokenizer
    
    all_messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": batch['img_path'][i]},
                    {"type": "text", "text": batch['prompt'][i]}
                ]
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text", item['gold_answer']}
            #     ]
            # }
            
        ]
    for i in range(len(batch['img_path']))]
    
    batch_text_inputs = processor.apply_chat_template(all_messages, tokenize=False, add_generation_prompt=True)
    batch_image_inputs, batch_video_inputs = process_vision_info(all_messages) 
    
    model_inputs = processor(
        text=batch_text_inputs,
        images=batch_image_inputs,
        videos=batch_video_inputs,
        return_tensors='pt',
        padding=True,
        truncation=False
    )
    model_inputs.to(device)

    # batch['model_inputs'] = model_inputs

    def _extract_spans(input_ids, start_token_id, end_token_id):
        start_positions = (input_ids == start_token_id).nonzero(as_tuple=True)[0]
        end_positions = (input_ids == end_token_id).nonzero(as_tuple=True)[0]
        spans = [(int(i.item()), int(j.item())) for i, j in zip(start_positions, end_positions)]
        return spans

    # Obtain marking spans
    batch_input_ids = model_inputs['input_ids']
    new_batch_input_ids = []
    batch_evidence_spans = []
    batch_attn_source_spans = []
    batch_attn_calibration_spans = []
    for input_ids in batch_input_ids:
        input_ids_tensor = torch.as_tensor(input_ids, dtype=torch.long)

        evidence_spans = _extract_spans(input_ids_tensor, tokenizer.convert_tokens_to_ids(EVIDENCE_START_TOKEN), tokenizer.convert_tokens_to_ids(EVIDENCE_END_TOKEN)) 
        attn_source_span = _extract_spans(input_ids_tensor, tokenizer.convert_tokens_to_ids(ATTN_SOURCE_START_TOKEN), tokenizer.convert_tokens_to_ids(ATTN_SOURCE_END_TOKEN))
        attn_calibration_span = _extract_spans(input_ids_tensor, tokenizer.convert_tokens_to_ids(ATTN_CALI_START_TOKEN), tokenizer.convert_tokens_to_ids(ATTN_CALI_END_TOKEN))

        batch_evidence_spans.append(evidence_spans)
        batch_attn_source_spans.append(attn_source_span)
        batch_attn_calibration_spans.append(attn_calibration_span)

        # print(f"DEBUG: evidence_spans: {tokenizer.decode(input_ids_tensor[evidence_spans[0][0]+1:evidence_spans[0][1]])}")
        # print(f"DEBUG: attn_source_span: {tokenizer.decode(input_ids_tensor[attn_source_span[0][0]+1:attn_source_span[0][1]])}")
        # print(f"DEBUG: attn_calibration_span: {tokenizer.decode(input_ids_tensor[attn_calibration_span[0][0]+1:attn_calibration_span[0][1]])}")
        # breakpoint()

        input_ids_tensor[input_ids_tensor == tokenizer.convert_tokens_to_ids(EVIDENCE_START_TOKEN)] = space_token_id
        input_ids_tensor[input_ids_tensor == tokenizer.convert_tokens_to_ids(EVIDENCE_END_TOKEN)] = space_token_id
        input_ids_tensor[input_ids_tensor == tokenizer.convert_tokens_to_ids(ATTN_SOURCE_START_TOKEN)] = space_token_id
        input_ids_tensor[input_ids_tensor == tokenizer.convert_tokens_to_ids(ATTN_SOURCE_END_TOKEN)] = space_token_id
        input_ids_tensor[input_ids_tensor == tokenizer.convert_tokens_to_ids(ATTN_CALI_START_TOKEN)] = space_token_id
        input_ids_tensor[input_ids_tensor == tokenizer.convert_tokens_to_ids(ATTN_CALI_END_TOKEN)] = space_token_id
        new_batch_input_ids.append(input_ids_tensor)

    model_inputs['input_ids'] = torch.stack(new_batch_input_ids)
    # batch['model_inputs']['input_ids'] = new_batch_input_ids
    # batch['evidence_spans'] = batch_evidence_spans
    # batch['attn_source_spans'] = batch_attn_source_spans
    # batch['attn_calibration_spans'] = batch_attn_calibration_spans

    # breakpoint()
    return model_inputs, batch_evidence_spans, batch_attn_source_spans, batch_attn_calibration_spans

    
def main(args):
    print("Attention span settings:")
    print(f"  Attn calibration span: {args.attn_calibration_span}")
    print(f"  Attn source span: {args.attn_source_span}")

    output_filepath = os.path.join(args.output_dir, f"attn_rerank_results-mode={args.attn_rank_mode}-n={args.take_n}.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(output_filepath) and args.use_cache:
        result_df = pd.read_csv(output_filepath)
    else:
        result_df = compute_attn_rerank_scores(args, output_filepath)
    
    base_df = pd.read_csv(args.base_inference_filepath)
    df = pd.merge(result_df, base_df, on='question_id', how='left')

    # compute metrics

    # def _hit_on_topk_batch(batch):
    #     gt_doc_ids = [item[0] for item in batch['pos_item_ids']]
    #     retrieved_doc_ids_list = [
    #         {x['passage_id'] for x in retrieved[:TOPK]}
    #         for retrieved in batch[RETRIEVAL_FIELD]
    #     ]
    #     hits = [
    #         gt_doc_id in retrieved_doc_ids
    #         for gt_doc_id, retrieved_doc_ids in zip(gt_doc_ids, retrieved_doc_ids_list)
    #     ]
    #     return {"hit": hits}
    
    # Since df is a pandas DataFrame, use apply instead of map
    # Assuming 'pos_item_ids' and 'retrieved_passage' columns exist and are lists of dicts
    import ast
    def _hit_on_topk_row(row):
        gt_doc_id = ast.literal_eval(row['pos_item_ids'])[0] if isinstance(row['pos_item_ids'], str) else row['pos_item_ids'][0]
        retrieved_ids = [ x['passage_id'] for x in ast.literal_eval(row[args.retrieval_field]) ] if isinstance(row[args.retrieval_field], str) else [ x['passage_id'] for x in row[args.retrieval_field] ]
        return gt_doc_id in retrieved_ids

    df['hit'] = df.apply(_hit_on_topk_row, axis=1)
    retriever_recall_at_5 = df['hit'].mean()
    vlm_recall_at_1 = df['hit_at_top1'].mean()
    
    # score given recalls
    score_given_attn_hit_top1 = df[df['hit_at_top1']==True]['score'].mean()
    overall_score = df['score'].mean()

    print(f"Retriever Recall@5: {retriever_recall_at_5}")
    print(f"VLM Attn Hit@1: {vlm_recall_at_1}")
    print(f"Score given VLM Attn Hit@1: {score_given_attn_hit_top1}")
    print(f"Overall Score: {overall_score}")
    print(f"Attn Prob on GT Doc: {df['attn_prob_on_gt_doc'].mean()}")

    # plot a bin diagram of x-axis=attn probability on gt doc; y-axis = score
    import matplotlib.pyplot as plt

    # Bin the attn_prob_on_gt_doc into 10 bins between 0 and 1
    num_bins = 10
    bins = np.linspace(0, 1, num_bins + 1)
    df['attn_bin'] = pd.cut(df['attn_prob_on_gt_doc'], bins, include_lowest=True)

    # Compute mean score for each bin
    bin_means = df.groupby('attn_bin')['score'].mean()
    bin_counts = df.groupby('attn_bin')['score'].count()
    bin_centers = [(interval.left + interval.right) / 2 for interval in bin_means.index]

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, bin_means, width=(1/num_bins)*0.9, align='center', alpha=0.7, edgecolor='k')
    plt.xlabel('Attn Probability on GT Doc (binned)')
    plt.ylabel('Mean Score')
    plt.title('Score vs. Attn Probability on GT Doc')
    plt.xticks(bin_centers, [f"{interval.left:.1f}-{interval.right:.1f}" for interval in bin_means.index], rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, "score_vs_attn_prob_on_gt_doc.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved score vs. attn probability plot to {plot_path}")

    
    

@torch.no_grad()
def compute_attn_rerank_scores(args, output_filepath):
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.add_special_tokens({
        "additional_special_tokens": [
            EVIDENCE_START_TOKEN, EVIDENCE_END_TOKEN,
            ATTN_SOURCE_START_TOKEN, ATTN_SOURCE_END_TOKEN,
            ATTN_CALI_START_TOKEN, ATTN_CALI_END_TOKEN
        ]
    })
    space_token_id = processor.tokenizer.encode(" ", add_special_tokens=False)[0]
    tokenize_func = partial(tokenize_and_get_spans, processor=processor, space_token_id=space_token_id)

    ds = load_from_disk(args.dataset_path)
    if args.ensure_passage_hit: # ensure that the ground truth document is in the topk retrieved documents
        def _hit_on_topk_batch(batch):
            gt_doc_ids = [item[0] for item in batch['pos_item_ids']]
            retrieved_doc_ids_list = [
                {x['passage_id'] for x in retrieved[:args.topk_docs]}
                for retrieved in batch[args.retrieval_field]
            ]
            hits = [
                gt_doc_id in retrieved_doc_ids
                for gt_doc_id, retrieved_doc_ids in zip(gt_doc_ids, retrieved_doc_ids_list)
            ]
            return {"hit": hits}

        ds = ds.map(_hit_on_topk_batch, batched=True, batch_size=250, num_proc=4)
        ds = ds.filter(lambda x: x['hit'])
        print(f"Filtered dataset to {len(ds)} instances, where the ground truth document is in the topk retrieved documents")
    if args.take_n is not None:
        ds = ds.select(range(args.take_n))
    
    passage_set, pid_to_content_map = load_passages(args.dataset_name)

    process_func = partial(add_prefix_and_form_prompt, args=args, pid_to_content_map=pid_to_content_map)
    ds = ds.map(process_func, batched=True, batch_size=args.process_batch_size, num_proc=1)


    # ds = ds.map(tokenize_func, batched=True, batch_size=args.process_batch_size, num_proc=1)
    def _collate_fn(batch):
        return {
            'img_path': [x['img_path'] for x in batch],
            'prompt': [x['prompt'] for x in batch],
            'gt_evidence_labels': [x['gt_evidence_labels'] for x in batch],
            'gold_answer': [x['gold_answer'] for x in batch],
            'question': [x['question'] for x in batch],
            'question_id': [x['question_id'] for x in batch],
            'question_type': [x['question_type'] for x in batch],
            'pos_item_ids': [x['pos_item_ids'] for x in batch],
            'answers': [x['answers'] for x in batch],
            args.retrieval_field: [x[args.retrieval_field][:args.topk_docs] for x in batch],
        }
    
    dataloader = DataLoader(ds, batch_size=args.forward_batch_size, shuffle=False, collate_fn=_collate_fn)
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, attn_implementation='eager')
    if args.lora_path is not None:
        model = PeftModel.from_pretrained(model, args.lora_path)
        print(f"Loaded LoRA adapter from {args.lora_path}")
    model.eval().to('cuda')

    attn_rank_scores = []
    rank_probs = []
    question_ids = []
    gt_doc_idx = []
    attn_top1_idx = []
    all_hit_top1 = []
    pos_item_ids = []
    retrieved_passages = []
    attn_prob_on_gt_doc = []
    for batch in tqdm(dataloader, desc="Processing batches", total=len(dataloader)):
        batch_size = len(batch['question_id'])
        model_inputs, batch_evidence_spans, batch_attn_source_spans, batch_attn_calibration_spans = tokenize_func(batch, device=model.device)
        forward_outputs = model(**model_inputs, return_dict=True, output_attentions=True)
        attention_weights = forward_outputs['attentions']
        for bidx in range(batch_size):
            scores = _compute_attn_reranking_scores(attention_weights, batch_attn_source_spans[bidx][0], batch_evidence_spans[bidx], bidx, args.attn_rank_mode)
            logits = scores
            log_probs = logits - torch.logsumexp(logits, dim=0)
            probs = torch.exp(log_probs)

            if 1 in batch['gt_evidence_labels'][bidx]:
                gt_idx = np.argmax(batch['gt_evidence_labels'][bidx])
                hit_top1 = (logits.argmax() == gt_idx).item()
            else:
                gt_idx = -1
                hit_top1 = False

            question_ids.append(batch['question_id'][bidx])
            attn_rank_scores.append(scores.detach().cpu().tolist())
            rank_probs.append(probs.detach().cpu().tolist())
            if 1 in batch['gt_evidence_labels'][bidx]:
                gt_doc_idx.append(batch['gt_evidence_labels'][bidx].index(1))
            else:
                gt_doc_idx.append(-1)
            attn_top1_idx.append(logits.argmax().item())
            all_hit_top1.append(hit_top1)
            pos_item_ids.append(batch['pos_item_ids'][bidx])
            retrieved_passages.append(batch[args.retrieval_field][bidx])
            attn_prob_on_gt_doc.append(probs[gt_idx].item() if gt_idx != -1 else -1)
            # print(f"DEBUG: document ranking scores: {scores}")
            # print(f"DEBUG: document ranking probabilities: {probs}")
            # print(f"DEBUG: ground truth passage index: {gt_idx}")
            # print(f"DEBUG: model top1 passage index: {logits.argmax().item()}")
            # print(f"DEBUG: attn prob on gt doc: {probs[gt_idx].item() if gt_idx != -1 else -1}")
            # print(f"DEBUG: hit top1: {hit_top1}")
            # breakpoint()

    result_df = pd.DataFrame({
        'question_id': question_ids,
        'gt_doc_idx': gt_doc_idx,
        'attn_top1_idx': attn_top1_idx,
        'attn_rank_scores': attn_rank_scores,
        'rank_probs': rank_probs,
        'hit_at_top1': all_hit_top1,
        'pos_item_ids': pos_item_ids,
        'attn_prob_on_gt_doc': attn_prob_on_gt_doc,
        args.retrieval_field: retrieved_passages
    })
    result_df.to_csv(output_filepath, index=False)
    print(f"Results saved to {output_filepath}")
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--img_basedir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="EVQA")
    parser.add_argument("--base_inference_filepath", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--take_n", type=int, default=None)
    parser.add_argument("--attn_rank_mode", type=str, required=True, choices=['sum', 'late-interaction'])
    parser.add_argument("--retrieval_field", type=str, default="retrieved_passage")
    parser.add_argument("--topk_docs", type=int, default=5)
    parser.add_argument("--process_batch_size", type=int, default=10)
    parser.add_argument("--forward_batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--attn_calibration_span", type=str, default='question_token', choices=['question_token'])
    parser.add_argument("--attn_source_span", type=str, default='question', choices=['question'])
    parser.add_argument("--use_cache", action='store_true')
    parser.add_argument("--ensure_passage_hit", action='store_true')
    args = parser.parse_args()
    main(args)