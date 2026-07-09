import sys
sys.path.append('src/')
from vqa_datasets import load_vqa_dataset
from qwen_vl_utils import process_vision_info
from vlms import QWen2VLM
import torch
import numpy as np

VLM_PROMPT_TEMPLATE = (
    "Answer the question after [QUESTION] about the image. A retriever has retrieved a relevant document for you and provided it after [EVIDENCE]. Give your answer after [ANSWER] without explanations\n"
)
# These two tokens are only used for marking the evidence span. They will be converted to ' ' (space) after identifying the span
EVIDENCE_START_TOKEN = "<evidence_start>"
EVIDENCE_END_TOKEN = "<evidence_end>"
TOPK_DOCS=5
ATTN_MODE = "eager" # "eager" or "flash_attention_2"

def _make_evidence(passage_dict):
    return (
        f"{EVIDENCE_START_TOKEN}"
        f"Title: {passage_dict['passage_id']}\t"
        f"Content: {passage_dict['text']}"
        f"{EVIDENCE_END_TOKEN}\n"
    )


def compute_attn_reranking_scores(attention_weights, from_spans, to_spans):
    # compute attention scores over all heads and all layers from `from_spans` to `to_spans`
    # attention_weights is (batch_size, num_heads, seq_len, seq_len)
    # from_spans is (batch_size, num_spans, 2)
    # to_spans is (batch_size, num_spans, 2)
    # return a tensor of shape (batch_size, num_spans)
    batch_size = attention_weights[0].shape[0]
    batch_scores = []
    
    for batch_idx in range(batch_size):
        from_start_idx, from_end_idx = from_spans[batch_idx][0]
        s = [] # (num of spans)
        
        for (to_start_idx, to_end_idx) in to_spans[batch_idx]:
            # Stack all layers into a single tensor: (num_layers, num_heads, from_span_len, to_span_len)
            layer_attn = torch.stack([
                attention_weights[l][batch_idx, :, from_start_idx:from_end_idx, to_start_idx:to_end_idx] 
                for l in range(len(attention_weights))
            ])
            
            # Sum over all layers and all heads in one operation
            ss = layer_attn.sum().detach().item()
            s.append(ss)
            
        batch_scores.append(s)
    return batch_scores

@torch.no_grad()
def main():
    ds = load_vqa_dataset("EVQA_with_evidence", split="test", take_n=10, img_basedir=".")
    vlm = QWen2VLM(
        model_path="QWen/QWen2-VL-2B-Instruct",
        processor_path="QWen/QWen2-VL-2B-Instruct",
        generation_config={
            'temperature': 0.3,
            'max_new_tokens': 64
        },
        attn_implementation=ATTN_MODE,
    )
    vlm._init_model()
    vocab_size = vlm.processor.tokenizer.vocab_size
    vlm.processor.tokenizer.add_special_tokens({
        "additional_special_tokens": [EVIDENCE_START_TOKEN, EVIDENCE_END_TOKEN],
    })
    # model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2VL-2B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
    attn_rerank_correct_count = 0
    for i, row in enumerate(ds):
        input_text = VLM_PROMPT_TEMPLATE
        evidence_passage_ids = []
        for i in range(TOPK_DOCS):
            input_text += f"[EVIDENCE] {_make_evidence(row['retrieved_passage'][i])}\n"
            evidence_passage_ids.append(row['retrieved_passage'][i]['passage_id'])
            if not row['pos_item_ids'][0] in evidence_passage_ids:
                gt_doc_item = {
                    'passage_id': row['pos_item_ids'][0],
                    'text': row['pos_item_contents']
                }
                input_text +=  _make_evidence(gt_doc_item) + "\n"
                evidence_passage_ids.append(row['pos_item_ids'][0])
        input_text += f"[QUESTION] {row['question']}"
        ans = f"[ANSWER] {row['gold_answer']}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": row['img_path']},
                    {"type": "text", "text": input_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ans}
                ]
            }
        ]
        text = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        before_ans_messages = messages[:-1]
        before_ans_text = vlm.processor.apply_chat_template(before_ans_messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = vlm.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Get the token ids for evidence start and end tokens
        evidence_start_token_id = vlm.processor.tokenizer(EVIDENCE_START_TOKEN, add_special_tokens=False).input_ids[0]
        evidence_end_token_id = vlm.processor.tokenizer(EVIDENCE_END_TOKEN, add_special_tokens=False).input_ids[0]

        # inputs['input_ids'] is (batch_size, seq_len)
        input_ids = inputs['input_ids']

        # Find all evidence spans for each example in the batch
        all_evidence_spans = []
        for batch_idx in range(input_ids.shape[0]):
            ids = input_ids[batch_idx]
            # Get all start and end positions
            start_positions = (ids == evidence_start_token_id).nonzero(as_tuple=True)[0]
            end_positions = (ids == evidence_end_token_id).nonzero(as_tuple=True)[0]

            # Pair up starts and ends in order (assume well-formed: each start has a matching end after it)
            spans = [(i+1, j) for i, j in zip(start_positions, end_positions)]
            all_evidence_spans.append(spans)

        # Print all evidence spans for each batch example
        for batch_idx, spans in enumerate(all_evidence_spans):
            print(f"Batch {batch_idx}:")
            for i, (start, end) in enumerate(spans):
                print(f"  Evidence span {i}: positions {start}-{end}. Text: {vlm.processor.tokenizer.decode(ids[start:end])}")

        before_ans_inputs = vlm.processor(
            text=[before_ans_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        attn_mask = inputs['attention_mask']
        before_ans_attn_mask = before_ans_inputs['attention_mask']
        before_ans_attn_mask = torch.nn.functional.pad(before_ans_attn_mask, (0, attn_mask.shape[-1] - before_ans_attn_mask.shape[-1]), "constant", False)

        # print(f"inputs.shape: {inputs['input_ids'].shape}")
        # print(f"before_ans_inputs.shape: {before_ans_inputs['input_ids'].shape}")
        # print(f"attn_mask.shape: {attn_mask.shape}")
        # print(f"before_ans_attn_mask.shape: {before_ans_attn_mask.shape}")
        # print(f"attn_mask[:-10]: {attn_mask[:,-10:]}")
        # print(f"before_ans_attn_mask[:-10]: {before_ans_attn_mask[:,-10:]}")
        # breakpoint()
        ans_position_mask = torch.logical_xor(attn_mask, before_ans_attn_mask)
        all_from_spans = []
        for batch_idx in range(ans_position_mask.shape[0]):
            from_spans = ans_position_mask[batch_idx].nonzero(as_tuple=True)[0]
            from_spans = [(from_spans[0], from_spans[-1])]
            all_from_spans.append(from_spans)

        # Before actual forward, replace EVIDENCE_START_TOKEN id and EVIDENCE_END_TOKEN id with the token id for " " (space)
        # Get the token id for " " (space)
        space_token_id = vlm.processor.tokenizer(" ", add_special_tokens=False).input_ids[0]

        # Replace EVIDENCE_START_TOKEN and EVIDENCE_END_TOKEN ids with space_token_id in input_ids
        input_ids = inputs['input_ids']
        input_ids[input_ids == evidence_start_token_id] = space_token_id
        input_ids[input_ids == evidence_end_token_id] = space_token_id

        if ATTN_MODE == "eager":
            forward_outputs = vlm.model(**inputs, return_dict=True, output_attentions=True)
            attention_weights = forward_outputs['attentions']
        elif ATTN_MODE == "flash_attention_2":
            forward_outputs = vlm.model(**inputs, return_dict=True, output_hidden_states=True, use_cache=True, output_attentions=True)
            print(f"{forward_outputs.keys()}")
            hidden_states = forward_outputs['hidden_states']
            breakpoint()
            attention_weights = compute_attn_weights_from_kv(hidden_state)
            for layer_idx in range(len(hidden_states)):
                layer_attn = hidden_states[layer_idx][:, :, :, :]
                attention_weights.append(layer_attn)
        else:
            raise ValueError(f"Unsupported attention mode: {ATTN_MODE}")

        print("all_from_spans: ", all_from_spans)
        print("all_evidence_spans: ", all_evidence_spans)
        all_attn_rerank_scores = compute_attn_reranking_scores(attention_weights, all_from_spans, all_evidence_spans)

        # Evaluation
        gt_doc_idx = evidence_passage_ids.index(row['pos_item_ids'][0])
        attn_rerank_highest_score_idx = np.argmax(all_attn_rerank_scores[0])
        for evidence_id, rerank_score in zip(evidence_passage_ids, all_attn_rerank_scores[0]):
            print(f"Evidence {evidence_id}: {rerank_score}")

        print(f"GT doc idx: {gt_doc_idx}")
        print(f"Attn rerank highest score idx: {attn_rerank_highest_score_idx}")
        if attn_rerank_highest_score_idx == gt_doc_idx:
            attn_rerank_correct_count += 1
            print(f"Attn rerank correct for example {i}")
        breakpoint()


if __name__ == "__main__":
    main()