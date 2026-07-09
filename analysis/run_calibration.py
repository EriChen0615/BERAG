"""
This code computes Expected Calibration Error (ECE) of a model for a given KBVQA dataset.
Input:
 - model
 - dataset
 - number of bins
 - number of candidate responses per question
 
Output:
 - candidate responses for each question
 - ECE value
 - calibration curve plot (confidence versus accuracy)

Example Usage:
"""

import argparse
import sys
sys.path.append('./src')
sys.path.append('./src/ops')
from vqa_datasets import load_vqa_dataset
from vlms import QWen2VLM
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import os
import json

BASE_PROMPT = "Answer the question about the image. A retriever has retrieved a relevant document for you and provided it after [EVIDENCE].\n\nQuestion:"

@torch.no_grad()
def sample_answer_candidates_from_model(vlm, input_text, input_img, num_sequences, temperature=0.1, seed=0):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": input_img},
                {"type": "text", "text": input_text},
            ],
        }
    ]

    text = vlm.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = vlm.model.generate(
        **inputs, 
        num_return_sequences=num_sequences,
        temperature=temperature,
        do_sample=True if temperature>0 else False,
        max_new_tokens=256,
        num_beams=1,
        top_k=50 if temperature > 0 else 1,
        top_p=0.95 if temperature > 0 else 1.0,
    )
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids.repeat_interleave(num_sequences, dim=0), generated_ids)]
    output_text = vlm.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text, 'sampled at temperature', temperature)
    return output_text
    
@torch.no_grad()
def compute_probs_from_vlm(vlm, input_text, input_img, ans_candidates):
    """
    This function computes the probabilities of the answer candidates from the VLM.
    i.e., P(ans | question, image, evidence) for ans in ans_candidates
    """
    answer_probs = []
    for ans in ans_candidates:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image", "image": input_img},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ans},
                ],
            }
        ]

        text = vlm.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = vlm.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # Get logits for the full sequence
        logits = vlm.model(**inputs).logits
        
        # Get the logits for generating the answer tokens
        # First get length of input without assistant's response
        messages_without_assistant = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image", "image": input_img},
                ],
            }
        ]
        text_without_assistant = vlm.processor.apply_chat_template(
            messages_without_assistant, tokenize=False, add_generation_prompt=True
        )
        inputs_without_assistant = vlm.processor(
            text=[text_without_assistant],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        answer_start_idx = len(inputs_without_assistant.input_ids[0])
        answer_tokens = vlm.processor.tokenizer.encode(ans, add_special_tokens=False)
        answer_logits = logits[0, answer_start_idx-1:answer_start_idx+len(answer_tokens)-1, :]

        # Calculate probabilities for each token
        probs = torch.softmax(answer_logits, dim=-1)
        
        # Get probability of generating each answer token
        answer_token_probs = []
        for i, token_id in enumerate(answer_tokens):
            answer_token_probs.append(probs[i, token_id].item())
            
        # Overall probability is product of token probabilities
        answer_prob = torch.prod(torch.tensor(answer_token_probs)).item()
        print("answer_prob for ", ans, "is", answer_prob)
        answer_probs.append(answer_prob)
    # Normalize probabilities to sum to 1
    answer_probs = torch.tensor(answer_probs)
    normalized_answer_probs = answer_probs / answer_probs.sum()
    return normalized_answer_probs.tolist(), answer_probs.tolist()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--is_lora", action="store_true")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--img_basedir", type=str, required=True)
    parser.add_argument("--take_n", type=int, required=True)
    parser.add_argument("--ds_seed", type=int, default=0)
    parser.add_argument("--num_candidate_responses", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_cases", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--answer_candidates_file", type=str, required=True)
    parser.add_argument("--evidence_field", type=str, default="retrieved_passage", choices=["retrieved_passage", "reranked_passage"])
    parser.add_argument("--topk_docs", type=int, default=1)
    parser.add_argument("--do_dpo_reward_selection", action="store_true")
    parser.add_argument("--ref_model_path", type=str, default=None)
    args = parser.parse_args()

    vqa_dataset = load_vqa_dataset(args.dataset_name, split=args.split, img_basedir=args.img_basedir, take_n=args.take_n, seed=args.ds_seed)
    if args.debug:
        if args.debug_cases:
            vqa_dataset = vqa_dataset.select([int(i) for i in args.debug_cases])
        else:
            vqa_dataset = vqa_dataset.select([i for i in range(10)])
    
    # Load model
    vlm = QWen2VLM(
        model_path=args.model_path,
        is_lora=args.is_lora,
        base_model_path=args.base_model_path,
        processor_path=args.processor_path,
        generation_config=None,
    )
    model_inited = False

    # Sample answer candidates
    question_id_to_answer_candidates_map = {}
    if os.path.exists(args.answer_candidates_file):
        question_id_to_answer_candidates_map = json.load(open(args.answer_candidates_file, 'r'))
    
    if args.evidence_field == "reranked_passage":
        from vqa_datasets import load_passages
        if 'InfoseekNew' in args.dataset_name:
            _, passage_id_to_text_map = load_passages("InfoseekNew_FullPassage", split='valid')
        elif 'EVQA' in args.dataset_name:
            _, passage_id_to_text_map = load_passages("EVQA", split='test')
        else:
            raise ValueError(f"Dataset {args.dataset_name} not supported")

    for i, item in enumerate(tqdm(vqa_dataset, total=len(vqa_dataset), desc="Inferencing")):
        question_id, question, img, gt_answer = item['question_id'], item['question'], item['img_path'], item['gold_answer']
        if question_id in question_id_to_answer_candidates_map:
            continue
        if args.evidence_field == "retrieved_passage":
            evidence = [' '.join(item[args.evidence_field][i]['text'].split(' ')[:1024]) for i in range(args.topk_docs)]
            evidence_id = [item[args.evidence_field][i]['passage_id'] for i in range(args.topk_docs)]
        elif args.evidence_field == "reranked_passage":
            evidence = [' '.join(passage_id_to_text_map[item[args.evidence_field][i]['passage_id']].split(' ')[:1024]) for i in range(args.topk_docs)]
            evidence_id = [item[args.evidence_field][i]['passage_id'] for i in range(args.topk_docs)]

        input_text = BASE_PROMPT + question + f"\n[EVIDENCE] {'[EVIDENCE] '.join(evidence)}" + f"\n[ANSWER] "
        ans_candidates = []
        # sample with temperature scaling
        # for temp in [0, 0.3, 0.5, 1.0, 1.5, 2.0]:
        if not model_inited:
            vlm._init_model()
            model_inited = True
        for temp in [0, 0.7]:
            ans_candidates += sample_answer_candidates_from_model(
                vlm, input_text, img, 
                num_sequences=1 if temp == 0 else args.num_candidate_responses, 
                temperature=temp,
                seed=args.seed, 
            )
            if temp == 0:
                greedy_response = ans_candidates[0]

        ans_candidates = list(set(ans_candidates)) # deduplicate
        print("unique answer candidates", ans_candidates)
        print("gt answer", gt_answer)
        question_id_to_answer_candidates_map[question_id] = {
            'question': question,
            'img_path': img,
            'evidence': evidence,
            'evidence_id': evidence_id,
            'answer_candidates': ans_candidates,
            'greedy_response': greedy_response,
            'gt_answer': gt_answer,
        }

    os.makedirs(os.path.dirname(args.answer_candidates_file), exist_ok=True)
    if not os.path.exists(args.answer_candidates_file):
        with open(args.answer_candidates_file, 'w') as f:
            json.dump(question_id_to_answer_candidates_map, f)

    # Mark candidate responses
    marked_filename = args.answer_candidates_file.replace('.json', '+marked.json')
    if os.path.exists(marked_filename):
        with open(marked_filename, 'r') as f:
            question_id_to_answer_candidates_map = json.load(f)
    else:
        if 'InfoseekNew' in args.dataset_name:
            for qid, item in question_id_to_answer_candidates_map.items():
                scores = []
                for ans in item['answer_candidates']:
                    if ans.lower() == item['gt_answer'].lower():
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
                item['scores'] = scores
        elif 'EVQA' in args.dataset_name:
            sys.path.append('./src/evaluation')
            from evqa_eval_1004 import process_row as eval_process_row
            from evqa_eval_1004 import process_row_mp as eval_process_row_mp
            import tensorflow as tf
            import pandas as pd

            df = vqa_dataset.to_pandas()
            qid_to_answer_candidates_df_data = [] 
            for qid, item in question_id_to_answer_candidates_map.items():
                qid_to_answer_candidates_df_data.append({
                    'question_id': qid,
                    'prediction': item['answer_candidates'],
                })
            
            to_merge_df = pd.DataFrame(qid_to_answer_candidates_df_data)
            eval_df = pd.merge(df, to_merge_df, on='question_id', how='left')

            eval_df = eval_df.explode('prediction')


            all_eval_results = []
            # if tf.test.is_gpu_available():
            if False: # default to CPU, as it's fast.
                for row in tqdm(eval_df.itertuples(), total=len(eval_df)):
                    eval_result = eval_process_row(row)
                    all_eval_results.append(eval_result)
                eval_df['score'] = all_eval_results
            else:
                import multiprocessing
                with multiprocessing.Pool(processes=4) as pool:
                    all_eval_results = list(tqdm(pool.imap(eval_process_row_mp, eval_df.iterrows(), chunksize=1), total=len(eval_df)))
                eval_df['score'] = all_eval_results
            # Group by question_id to collect all predictions and scores
            grouped_df = eval_df.groupby('question_id').agg({
                'prediction': list,
                'score': list
            }).reset_index()

            # Update the answer candidates map with scores
            for _, row in grouped_df.iterrows():
                qid = row['question_id']
                predictions = row['prediction'] 
                scores = row['score']
                
                # Update the scores in the original map
                if qid in question_id_to_answer_candidates_map:
                    # Verify predictions match answer candidates
                    question_id_to_answer_candidates_map[qid]['answer_candidates'] = predictions
                    question_id_to_answer_candidates_map[qid]['scores'] = scores
        else:
            raise NotImplementedError(f"Dataset {args.dataset_name} not supported for evaluation")

        with open(marked_filename, 'w') as f:
            json.dump(question_id_to_answer_candidates_map, f)

    prob_filename = marked_filename.replace('.json', '+probs.json')
    if os.path.exists(prob_filename):
        with open(prob_filename, 'r') as f:
            question_id_to_answer_candidates_map = json.load(f)
    else:
        for qid, item in tqdm(question_id_to_answer_candidates_map.items(), total=len(question_id_to_answer_candidates_map), desc="Computing probabilities"):
            question, img, evidence = item['question'], item['img_path'], '[EVIDENCE] '.join(item['evidence'])
            input_text = BASE_PROMPT + question + f"\n[EVIDENCE] {evidence}" + f"\n[ANSWER] "
            ans_candidates = item['answer_candidates']
            # sample with temperature scaling
            if not model_inited:
                vlm._init_model()
                model_inited = True
            model_probs, raw_model_probs = compute_probs_from_vlm(vlm, input_text, img, ans_candidates)
            item['model_probs'] = model_probs
            item['raw_model_probs'] = raw_model_probs
        with open(prob_filename, 'w') as f:
            json.dump(question_id_to_answer_candidates_map, f)
    
    # Now, we can compute ECE and plot calibration curve. Let's start with the calibration curve.
    # Compute calibration curve
    import numpy as np
    import matplotlib.pyplot as plt
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Compute calibration curve
    confidences = []
    accuracies = []
    
    # Collect all probabilities and scores
    all_probs = []
    all_scores = []
    for qid, item in question_id_to_answer_candidates_map.items():
        # Filter out low probability candidates (<0.01), considering our bin width is 0.1. 
        for prob, score in zip(item['raw_model_probs'], item['scores']):
            if prob >= 0.01:
                all_probs.append(prob)
                all_scores.append(score)
    all_probs = np.array(all_probs)
    all_scores = np.array(all_scores)
    
    # Compute average accuracy for each bin
    for i in range(num_bins):
        bin_lower = bin_lowers[i]
        bin_upper = bin_uppers[i]
        
        # Find all predictions that fall into this bin
        bin_indices = (all_probs >= bin_lower) & (all_probs < bin_upper)
        if np.any(bin_indices):
            bin_accuracy = np.mean(all_scores[bin_indices])
            # Use bin center as x-coordinate
            bin_center = (bin_lower + bin_upper) / 2
            confidences.append(bin_center)
            accuracies.append(bin_accuracy)
    
    # Plot calibration curve as bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(confidences, accuracies, width=1/num_bins, align='center', label='Calibration Curve', alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', marker='o', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curve (filtered p>=0.05)')
    plt.legend()
    plt.savefig(args.answer_candidates_file.replace('.json', '+calibration_curve.png'))
    plt.show()

    # Now, compute ECE = ∑bi||(pi-ci)|| where bi is the number of samples in bin i, pi is the average confidence in bin i, and ci is the average accuracy in bin i.
    ece = 0
    for i in range(num_bins):
        bin_lower = bin_lowers[i]
        bin_upper = bin_uppers[i]
        bin_indices = (all_probs >= bin_lower) & (all_probs < bin_upper)
        # Skip empty bins to avoid nan from np.mean on empty arrays
        if np.any(bin_indices):
            bin_accuracy = np.mean(all_scores[bin_indices])
            bin_confidence = np.mean(all_probs[bin_indices])
            ece += np.abs(bin_confidence - bin_accuracy) * np.sum(bin_indices) / len(all_probs)

    print(f"Expected Calibration Error (ECE) = {ece}")
    
    # Get some performance metrics (e.g., acc@MaxProb, acc@Oracle)
    # Calculate acc@MaxProb - accuracy when selecting most probable answer for each question
    max_prob_correct = 0
    oracle_correct = 0
    greedy_correct = 0
    total_questions = 0
    avg_log_gain = 0
    for qid, item in question_id_to_answer_candidates_map.items():
        probs = item['model_probs']
        scores = item['scores']
        
        # Get prediction from highest probability
        max_prob_idx = np.argmax(probs)
        if scores[max_prob_idx] == 1:
            max_prob_correct += 1
            
        # Get oracle prediction (highest scoring answer)
        oracle_idx = np.argmax(scores)
        if scores[oracle_idx] == 1:
            oracle_correct += 1
        
        greedy_idx = item['answer_candidates'].index(item['greedy_response'])
        if scores[greedy_idx] == 1:
            greedy_correct += 1
            
        total_questions += 1

    
        
    acc_max_prob = max_prob_correct / total_questions
    acc_oracle = oracle_correct / total_questions
    acc_greedy = greedy_correct / total_questions
    print(f"Accuracy when selecting highest probability answer (acc@MaxProb) = {acc_max_prob:.3f}")
    print(f"Accuracy when selecting highest scoring answer (acc@Oracle) = {acc_oracle:.3f}")
    print(f"Accuracy when selecting greedy answer (acc@Greedy) = {acc_greedy:.3f}")
    
    
    acc_dpo = -1
    avg_cpmf = -100
    avg_cpmf_oracle_correct = -1
    avg_cpmf_oracle_wrong = -1
    if args.do_dpo_reward_selection:
        ref_prob_filename = args.answer_candidates_file.replace('.json', '+ref_probs.json')
        if os.path.exists(ref_prob_filename):
            with open(ref_prob_filename, 'r') as f:
                question_id_to_answer_candidates_map = json.load(f)
        else:
            if model_inited:
                vlm = None
                # Initialize reference model
            ref_vlm = QWen2VLM(
                model_path=args.ref_model_path,
                is_lora=False, # reference model cannot be a LoRA model
                processor_path=args.processor_path,
                generation_config=None,
            )
            ref_vlm._init_model()

            # Compute reference probabilities and DPO rewards
            
            
            for qid, item in tqdm(question_id_to_answer_candidates_map.items(), total=len(question_id_to_answer_candidates_map), desc="Computing reference probabilities"):
                question, img, evidence = item['question'], item['img_path'], '[EVIDENCE] '.join(item['evidence'])
                input_text = BASE_PROMPT + question + f"\n[EVIDENCE] {evidence}" + f"\n[ANSWER] "
                ans_candidates = item['answer_candidates']
                
                # Get reference model probabilities
                ref_model_probs, ref_raw_model_probs = compute_probs_from_vlm(ref_vlm, input_text, img, ans_candidates)
                item['ref_model_probs'] = ref_model_probs
                item['ref_raw_model_probs'] = ref_raw_model_probs
                
                # Compute DPO rewards
                policy_probs = np.array(item['raw_model_probs'])
                ref_probs = np.array(ref_raw_model_probs)
                dpo_rewards = np.log(policy_probs) - np.log(ref_probs)
                
              
                
                # Store DPO rewards
                item['dpo_rewards'] = dpo_rewards.tolist()
            
            with open(ref_prob_filename, 'w') as f:
                json.dump(question_id_to_answer_candidates_map, f)
        
            ref_vlm = None

        # Compute DPO accuracy after having loaded/computed the probabilities
        dpo_correct = 0
        total_questions = 0
        for qid, item in question_id_to_answer_candidates_map.items():
            # Select answer with highest DPO reward
            dpo_rewards = np.array(item['dpo_rewards'])
            dpo_idx = np.argmax(dpo_rewards)
            if item['scores'][dpo_idx] == 1:
                dpo_correct += 1
            total_questions += 1
        
        acc_dpo = dpo_correct / total_questions
        print(f"Accuracy when selecting by DPO reward = {acc_dpo:.3f}")

       
            
        # Calculate net correct probability mass flow (△cpmf|) as follows
        # Let pi be the probability assigned by the policy model to answer ai. Let qi be the probability assigned by the reference model to answer ai
        # Let yi (1, -1) be the indicator for whether ai is correct or wrong.
        # Define △cpmf = ∑_i yi(pi - qi). Use the model's raw probabilities to compute this.
        # Calculate net correct probability mass flow (△cpmf)
        total_cpmf = 0
        for qid, item in question_id_to_answer_candidates_map.items():
            policy_probs = np.array(item['raw_model_probs'])
            ref_probs = np.array(item['ref_raw_model_probs'])
            scores = np.array(item['scores'])
            
            # Convert 0/1 scores to -1/1
            y_i = 2*scores - 1
            
            # Calculate probability mass flow for this question
            prob_mass_flow = np.sum(y_i * (policy_probs - ref_probs))
            total_cpmf += prob_mass_flow
        
        # Calculate △cpmf under two conditions. 
        # (1) cpmf_oracle_correct. when the ground-truth answer is in one of the responses.
        # (2) cpmf_oracle_wrong. when the ground-truth answer is not in one of the resopnses. 
        # Average across all questions
        # Calculate △cpmf separately for oracle correct/wrong cases
        cpmf_oracle_correct = 0
        cpmf_oracle_wrong = 0
        num_oracle_correct = 0
        num_oracle_wrong = 0
        total_log_cpmf = 0
        
        for qid, item in question_id_to_answer_candidates_map.items():
            policy_probs = np.array(item['raw_model_probs'])
            ref_probs = np.array(item['ref_raw_model_probs'])
            scores = np.array(item['scores'])
            
            # Convert 0/1 scores to -1/1
            y_i = 2*scores - 1
            
            # Calculate probability mass flow for this question
            prob_mass_flow = np.sum(y_i * (policy_probs - ref_probs))
            
            # Check if oracle was correct (any score=1)
            if np.any(scores == 1):
                cpmf_oracle_correct += prob_mass_flow
                num_oracle_correct += 1
            
            else:
                cpmf_oracle_wrong += prob_mass_flow
                num_oracle_wrong += 1
            
            
        # Calculate averages, handling division by zero
        avg_cpmf_oracle_correct = cpmf_oracle_correct / num_oracle_correct if num_oracle_correct > 0 else 0
        avg_cpmf_oracle_wrong = cpmf_oracle_wrong / num_oracle_wrong if num_oracle_wrong > 0 else 0
        
        print(f"Average △cpmf when oracle correct = {avg_cpmf_oracle_correct:.3f}")
        print(f"Average △cpmf when oracle wrong = {avg_cpmf_oracle_wrong:.3f}")
        avg_cpmf = total_cpmf / len(question_id_to_answer_candidates_map)
        print(f"Average correct probability mass flow (△cpmf) = {avg_cpmf:.3f}")

        # Now let's compute average pairwise log-probability gain as dictated by the ideal dpo equaiton
        # Iterate all pairs for which the ground-truth answer is in one of the response. For each pair, 
        # Let pw be the probability of the winning response, and let pl be the probability of the losing response.
        # Pairwise-LogGain (P-LG) = 1/N ∑_(w,l) log(pw)-log(pl)
        # Ideal DPO dictates that this should be  P-LG = 1/N ∑_(w,l) log(qw)-log(ql)+1/beta CONST(true preference probability)
        # Calculate pairwise log probability gain
        total_pairwise_log_gain = 0
        total_winning_log_gain = 0
        num_pairs = 0
        num_winning_resps = 0

        for qid, item in question_id_to_answer_candidates_map.items():
            policy_probs = np.array(item['raw_model_probs'])
            scores = np.array(item['scores'])
            
            # Only consider cases where ground truth is present
            if not np.any(scores == 1):
                continue
                
            # Get indices of winning (score=1) and losing (score=0) responses
            winning_indices = np.where(scores == 1)[0]
            losing_indices = np.where(scores == 0)[0]
            
            # Calculate log gain for all winning-losing pairs
            for w_idx in winning_indices:
                for l_idx in losing_indices:
                    pw = policy_probs[w_idx]
                    pl = policy_probs[l_idx]
                    
                    # Add small epsilon to avoid log(0)
                    eps = 1e-10
                    log_gain = np.log(pw + eps) - np.log(pl + eps)
                    
                    total_pairwise_log_gain += log_gain
                    num_pairs += 1
            
            # Calculate the log gain for all winning responses under the policy model compared to the reference model
            # Calculate log gain between policy and reference model probabilities for winning responses
            ref_probs = np.array(item['ref_raw_model_probs'])
            for w_idx in winning_indices:
                pw_policy = policy_probs[w_idx]
                pw_ref = ref_probs[w_idx]
                
                # Add small epsilon to avoid log(0)
                eps = 1e-10
                log_gain = np.log(pw_policy + eps) - np.log(pw_ref + eps)
                
                total_winning_log_gain += log_gain
                num_winning_resps += 1
    
        # Calculate average
        avg_pairwise_log_gain = total_pairwise_log_gain / num_pairs if num_pairs > 0 else 0
        avg_winning_log_gain = total_winning_log_gain / num_winning_resps if num_winning_resps > 0 else 0
        print(f"Average pairwise log probability gain = {avg_pairwise_log_gain:.3f}")
        print(f"Average winning log probability gain = {avg_winning_log_gain:.3f}")


    stats_filename = args.answer_candidates_file.replace('.json', '+stats.json')
    with open(stats_filename, 'w') as f:
        json.dump({
            'ece': ece,
            'acc_max_prob': acc_max_prob,
            'acc_oracle': acc_oracle,
            'acc_dpo': acc_dpo,
            'acc_greedy': acc_greedy,
            'avg_cpmf': avg_cpmf,
            'avg_cpmf_oracle_correct': avg_cpmf_oracle_correct,
            'avg_cpmf_oracle_wrong': avg_cpmf_oracle_wrong,
            'avg_pairwise_log_gain': avg_pairwise_log_gain,
            'avg_winning_log_gain': avg_winning_log_gain,
        }, f)