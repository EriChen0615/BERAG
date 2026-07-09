import json
import pandas as pd
from datasets import Dataset
import os
from collections import defaultdict
from tqdm import tqdm
import sys
import numpy as np
import random
from transformers import AutoTokenizer

# DATASET_NAME = "OKVQA"
# SAMPLE_EXPDIRS = [
#     "outputs/20241211-18-OKVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L_SEED=42",
#     "outputs/20241211-18-OKVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L_SEED=1129",
#     "outputs/20241211-21-OKVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L_SEED=2313",
#     "outputs/20241211-21-OKVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L_SEED=122423",
#     "outputs/20241211-21-OKVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L_SEED=615926",
#     "outputs/20241211-22-OKVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L_SEED=0",
#     "outputs/20241211-22-OKVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L_SEED=53",
#     "outputs/20241211-22-OKVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L_SEED=2341",
# ]
# TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}.csv"

# IMG_BASEDIR = '../..' # relative to third_party/LlamaFactory

DATASET_NAME = "EVQA"
SAMPLE_EXPDIRS = [
    "outputs/20241211-18-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=42",
    "outputs/20241211-18-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=1129",
    "outputs/20241211-19-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=0",
    "outputs/20241211-19-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=53",
    "outputs/20241211-19-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=2313",
    "outputs/20241211-19-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=2341",
    "outputs/20241211-19-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=122423",
    "outputs/20241211-19-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=615926",
]
# # TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}.csv"
# SAMPLE_EXPDIRS = [
#     "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=0",
#     "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=42",
#     "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=53",
#     "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=1129",
#     "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=2313",
#     "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=2341",
#     "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=122423",
#     "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=615926",
# ]

# TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_QWen2-VL-7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}.csv"

IMG_BASEDIR = '../..' # relative to third_party/LlamaFactory

# DATASET_NAME="Infoseek"
# SAMPLE_EXPDIRS = [
#     "outputs/20241224-14-Infoseek_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=0",
#     "outputs/20241226-15-Infoseek_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=0",
#     "outputs/20241226-15-Infoseek_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=42",
#     "outputs/20241226-15-Infoseek_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=53",
#     # "outputs/20241226-15-Infoseek_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=1129",
#     "outputs/20241226-15-Infoseek_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=2313",
#     "outputs/20241226-15-Infoseek_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=2341",
#     "outputs/20241226-15-Infoseek_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=122423",
#     "outputs/20241226-15-Infoseek_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=615926",
# ]

# DATASET_NAME = "InfoseekNew"
# SAMPLE_EXPDIRS = [
#     "outputs/20250128-12-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top5_PreFLMR-L_SEED=42",
#     "outputs/20250128-12-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top5_PreFLMR-L_SEED=2313",
#     "outputs/20250128-12-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top5_PreFLMR-L_SEED=122423",
#     "outputs/20250128-12-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top5_PreFLMR-L_SEED=615926",
#     "outputs/20250128-13-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top5_PreFLMR-L_SEED=0",
#     "outputs/20250128-13-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top5_PreFLMR-L_SEED=53",
#     "outputs/20250128-13-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top5_PreFLMR-L_SEED=1214",
#     "outputs/20250128-13-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top5_PreFLMR-L_SEED=2341",
# ]

# DATASET_NAME = "InfoseekNew"
# SAMPLE_EXPDIRS = [
#     "outputs/20250226-02-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=42",
#     "outputs/20250226-03-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=122423",
#     "outputs/20250226-04-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=0",
#     "outputs/20250226-04-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=53",
#     "outputs/20250226-04-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=2313",
#     "outputs/20250226-04-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=2341",
#     "outputs/20250226-04-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=615926",
#     "outputs/20250226-05-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top1_PreFLMR-L_SEED=1214",
# ]

# DATASET_NAME = "InfoseekNew"
# SAMPLE_EXPDIRS = [
#     "outputs/20250226-05-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top3_PreFLMR-L_SEED=42",
#     "outputs/20250226-05-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top3_PreFLMR-L_SEED=122423",
#     "outputs/20250226-06-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top3_PreFLMR-L_SEED=0",
#     "outputs/20250226-06-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top3_PreFLMR-L_SEED=53",
#     "outputs/20250226-06-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top3_PreFLMR-L_SEED=2313",
#     "outputs/20250226-06-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top3_PreFLMR-L_SEED=2341",
#     "outputs/20250226-06-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top3_PreFLMR-L_SEED=615926",
#     "outputs/20250226-07-InfoseekNew_train-64000_SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-Top3_PreFLMR-L_SEED=1214",
# ]

# TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_QWen2-VL-7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}.csv"
# TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_QWen2-VL-7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}_rag1.csv"
# TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_QWen2-VL-7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}_rag3.csv"

# IMG_BASEDIR = '' # relative to third_party/LlamaFactory
K_RESPONSES_PER_PAIR = 4
TARGET_DATA_FILE = f"third_party/LLaMAFactory/data/jinghong_chen/evqa/7B-rag5-answer-kto_max=4096/train_sharegpt_augmented_add_k={K_RESPONSES_PER_PAIR}.json"

def read_evqa_samples(expdir, qid_answer_map):
    with open(os.path.join(expdir, "histories.json")) as f:
        histories = json.load(f)
    with open(os.path.join(expdir, "VLMReadEvidence_histories.json")) as f:
        vlm_read_histories = json.load(f)
    marked_df = pd.read_csv(os.path.join(expdir, "marked_answers.csv"))

    for history, vlm_history, df_item in tqdm(zip(histories, vlm_read_histories, marked_df.itertuples()), total=len(histories), desc=f"Reading from {expdir}"):
        question_id = history[0][0]['question_id']
        img_path = os.path.join(IMG_BASEDIR, history[0][0]['img_path'])
        answer = history[-1][0].split('[ANSWER] ')[-1]
        score = df_item.score
        assert question_id == df_item.question_id

        qid_answer_map[question_id]['answers'].append(answer)
        qid_answer_map[question_id]['scores'].append(score)
        qid_answer_map[question_id]['img_path'] = img_path
        qid_answer_map[question_id]['gt_answer'] = df_item.gold_answer
        qid_answer_map[question_id]['prompt'] = vlm_history[3]['input_text']

if __name__ == '__main__':
    qid_answer_map = defaultdict(lambda: defaultdict(list))

    for expdir in SAMPLE_EXPDIRS:
        read_evqa_samples(expdir, qid_answer_map)

    # Initialize tokenizer for length checking
    tokenizer = AutoTokenizer.from_pretrained("QWen/QWen2-VL-2B-Instruct")
    
    kto_data = []
    good_cnt = 0
    bad_cnt = 0
    filtered_cnt = 0
    
    # flatten data
    for question_id, item in tqdm(qid_answer_map.items(), total=len(qid_answer_map), desc='flattening data...'):
        answers, scores, prompt, img_path, gt_answer = item['answers'], item['scores'], item['prompt'], item['img_path'], item['gt_answer']
        incorrect_answers = [ans for ans, s in zip(answers, scores) if s < 0.6]

        max_idx = np.argmax(scores)
        best_answer, best_score, mean_score = answers[max_idx], scores[max_idx], np.mean(scores)

        has_correct = True
        if best_score < 0.6:
            best_answer = gt_answer
            has_correct = False

        if len(incorrect_answers) == 0: # all correct; skip
            continue

        # Check input length before adding examples
        base_prompt = "<image> " + prompt
        base_length = len(tokenizer(base_prompt).input_ids)
        
        # Calculate max response length
        max_response_length = len(tokenizer(best_answer).input_ids)
        
        # Skip if total length would exceed 4096
        if base_length + max_response_length > 4096:
            filtered_cnt += 1
            continue

        # Always include best answer and one random incorrect answer
        k = random.randint(0, len(incorrect_answers)-1)
        conversations = [
            {
                "id": f"evqa_{question_id}",
                "conversations": [
                    {"from": "human", "value": base_prompt},
                    {"from": "gpt", "value": best_answer},
                ],
                "images": [img_path],
                "kto_tag": True
            },
            {
                "id": f"evqa_{question_id}",
                "conversations": [
                    {"from": "human", "value": base_prompt},
                    {"from": "gpt", "value": incorrect_answers[k]},
                ],
                "images": [img_path],
                "kto_tag": False
            }
        ]
        good_cnt += 1
        bad_cnt += 1

        # Add additional responses up to K_RESPONSES_PER_PAIR
        existing_answers = {best_answer, incorrect_answers[k]}
        cur_k = 2
        for ans, s in zip(answers, scores):
            if ans not in existing_answers and cur_k <= K_RESPONSES_PER_PAIR:
                # Check length for this specific answer
                if base_length + len(tokenizer(ans).input_ids) > 4096:
                    continue
                    
                is_good = s >= 0.6
                conversations.append({
                    "id": f"evqa_{question_id}",
                    "conversations": [
                        {"from": "human", "value": base_prompt},
                        {"from": "gpt", "value": ans},
                    ],
                    "images": [img_path],
                    "kto_tag": is_good
                })
                existing_answers.add(ans)
                cur_k += 1
                if is_good:
                    good_cnt += 1
                else:
                    bad_cnt += 1

        kto_data.extend(conversations)

    print(f"Filtered out {filtered_cnt} examples exceeding 4096 tokens")
    print(f"Total good responses: {good_cnt}")
    print(f"Total bad responses: {bad_cnt}")
    print(f"Good/Bad ratio: {good_cnt/bad_cnt:.2f}")

    os.makedirs(os.path.dirname(TARGET_DATA_FILE), exist_ok=True)
    with open(TARGET_DATA_FILE, 'w') as f:
        json.dump(kto_data, f, indent=2)
    print(f"KTO data saved to {TARGET_DATA_FILE}")


