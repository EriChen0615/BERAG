import json
import pandas as pd
from datasets import Dataset
import os
from collections import defaultdict
from tqdm import tqdm
import sys
import numpy as np
import random

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

# DATASET_NAME = "EVQA"
# # SAMPLE_EXPDIRS = [
# #     "outputs/20241211-18-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=42",
# #     "outputs/20241211-18-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=1129",
# #     "outputs/20241211-19-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=0",
# #     "outputs/20241211-19-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=53",
# #     "outputs/20241211-19-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=2313",
# #     "outputs/20241211-19-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=2341",
# #     "outputs/20241211-19-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=122423",
# #     "outputs/20241211-19-EVQA_train-0_SampleCacheRetrieve[Top5]-Read_QWen2VL-2B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=615926",
# # ]
# # TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}.csv"
SAMPLE_EXPDIRS = [
    "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=0",
    "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=42",
    "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=53",
    "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=1129",
    "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=2313",
    "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=2341",
    "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=122423",
    "outputs/20241217-15-EVQA_train-64000_SampleCacheRetrieve[Top5]-Read_QWen2VL-7B-LoRA_RAG-Top1_PreFLMR-L-train_SEED=615926",
]

TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_QWen2-VL-7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}.csv"
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

# # TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_QWen2-VL-7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}.csv"
# # TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_QWen2-VL-7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}_rag1.csv"
# TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_QWen2-VL-7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}_rag3.csv"

# DATASET_NAME = "EVQA"
# SAMPLE_EXPDIRS = [
#     "outputs/20250306-08-EVQA_train-64000_SampleCacheRetrieve[Top3]-Read_QWen2VL-7B-LoRA_RAG-Top1_basemodel_PreFLMR-L-train_SEED=0",
#     "outputs/20250306-08-EVQA_train-64000_SampleCacheRetrieve[Top3]-Read_QWen2VL-7B-LoRA_RAG-Top1_basemodel_PreFLMR-L-train_SEED=42",
#     "outputs/20250306-08-EVQA_train-64000_SampleCacheRetrieve[Top3]-Read_QWen2VL-7B-LoRA_RAG-Top1_basemodel_PreFLMR-L-train_SEED=53",
#     "outputs/20250306-08-EVQA_train-64000_SampleCacheRetrieve[Top3]-Read_QWen2VL-7B-LoRA_RAG-Top1_basemodel_PreFLMR-L-train_SEED=1129",
#     "outputs/20250306-08-EVQA_train-64000_SampleCacheRetrieve[Top3]-Read_QWen2VL-7B-LoRA_RAG-Top1_basemodel_PreFLMR-L-train_SEED=2313",
#     "outputs/20250306-08-EVQA_train-64000_SampleCacheRetrieve[Top3]-Read_QWen2VL-7B-LoRA_RAG-Top1_basemodel_PreFLMR-L-train_SEED=2341",
#     "outputs/20250306-08-EVQA_train-64000_SampleCacheRetrieve[Top3]-Read_QWen2VL-7B-LoRA_RAG-Top1_basemodel_PreFLMR-L-train_SEED=122423",
#     "outputs/20250306-08-EVQA_train-64000_SampleCacheRetrieve[Top3]-Read_QWen2VL-7B-LoRA_RAG-Top1_basemodel_PreFLMR-L-train_SEED=615926",
# ]
# TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}_rag3.csv"
# IMG_BASEDIR = '../..' # relative to third_party/LlamaFactory

# DATASET_NAME = "EVQA"
# SAMPLE_EXPDIRS = [
#     "outputs/20250306-08-EVQA_train-64000_SampleCacheRetrieve[Top1]-Read_QWen2VL-7B-LoRA_RAG-Top1-basemodel_PreFLMR-L-train_SEED=42",
#     "outputs/20250306-08-EVQA_train-64000_SampleCacheRetrieve[Top1]-Read_QWen2VL-7B-LoRA_RAG-Top1-basemodel_PreFLMR-L-train_SEED=1129",
#     "outputs/20250306-08-EVQA_train-64000_SampleCacheRetrieve[Top1]-Read_QWen2VL-7B-LoRA_RAG-Top1-basemodel_PreFLMR-L-train_SEED=122423",
#     "outputs/20250306-09-EVQA_train-64000_SampleCacheRetrieve[Top1]-Read_QWen2VL-7B-LoRA_RAG-Top1-basemodel_PreFLMR-L-train_SEED=0",
#     "outputs/20250306-09-EVQA_train-64000_SampleCacheRetrieve[Top1]-Read_QWen2VL-7B-LoRA_RAG-Top1-basemodel_PreFLMR-L-train_SEED=53",
#     "outputs/20250306-09-EVQA_train-64000_SampleCacheRetrieve[Top1]-Read_QWen2VL-7B-LoRA_RAG-Top1-basemodel_PreFLMR-L-train_SEED=2313",
#     "outputs/20250306-09-EVQA_train-64000_SampleCacheRetrieve[Top1]-Read_QWen2VL-7B-LoRA_RAG-Top1-basemodel_PreFLMR-L-train_SEED=2341",
#     "outputs/20250306-09-EVQA_train-64000_SampleCacheRetrieve[Top1]-Read_QWen2VL-7B-LoRA_RAG-Top1-basemodel_PreFLMR-L-train_SEED=615926",
# ]

# TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}_rag1.csv"
# IMG_BASEDIR = '../..' # relative to third_party/LlamaFactory

# DATASET_NAME = "InfoseekNew"
# SAMPLE_EXPDIRS = [
#     "outputs/20250317-16-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=42",
#     "outputs/20250317-16-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=122423",
#     "outputs/20250317-16-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=615926",
#     "outputs/20250317-17-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=53",
#     "outputs/20250317-17-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=2313",
#     "outputs/20250317-17-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=2341",
#     "outputs/20250317-18-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=0",
#     "outputs/20250317-18-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=1214",
# ]

# IMG_BASEDIR = ""
# TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_QWen2-VL-7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}_rag1_it2.csv"

# DATASET_NAME = "InfoseekNew"
# SAMPLE_EXPDIRS = [
#     "outputs/20250423-23-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=42",
#     "outputs/20250423-23-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=53",
#     "outputs/20250423-23-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=2313",
#     "outputs/20250423-23-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=2341",
#     "outputs/20250423-23-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=122423",
#     "outputs/20250423-23-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=615926",
#     "outputs/20250424-00-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=0",
#     "outputs/20250424-00-InfoseekNew_train-64000_2nd-SampleCacheRetrieve[TopK]-Read_QWen2VL-7B_RAG-DPO-Top1_PreFLMR-L_SEED=1214"
# ]   

# IMG_BASEDIR = ""
# TARGET_FILENAME = f"data/jinghong_chen/{DATASET_NAME}_QWen2-VL-7B_self-generated-pairs_n={len(SAMPLE_EXPDIRS)}_rag1_it2-0424.csv"


def read_infoseek_samples(expdir, qid_answer_map):
    with open(os.path.join(expdir, "histories.json")) as f:
        histories = json.load(f)
    with open(os.path.join(expdir, "VLMReadEvidence_histories.json")) as f:
        vlm_read_histories = json.load(f)

    for history, vlm_history in tqdm(zip(histories, vlm_read_histories), total=len(histories), desc=f"Reading from {expdir}"):
        question_id = history[0][0]['question_id']
        img_path = os.path.join(IMG_BASEDIR, history[0][0]['img_path'])
        answer = history[-1][0].split('[ANSWER] ')[-1]

        qid_answer_map[question_id]['answers'].append(answer)
        qid_answer_map[question_id]['img_path'] = img_path
        qid_answer_map[question_id]['prompt'] = vlm_history[3]['input_text']


def read_okvqa_samples(expdir, qid_answer_map):
    with open(os.path.join(expdir, "histories.json")) as f:
        histories = json.load(f)
    with open(os.path.join(expdir, "VLMReadEvidence_histories.json")) as f:
        vlm_read_histories = json.load(f)

    for history, vlm_history in tqdm(zip(histories, vlm_read_histories), total=len(histories), desc=f"Reading from {expdir}"):
        question_id = history[0][0]['question_id']
        img_path = os.path.join(IMG_BASEDIR, history[0][0]['img_path'])
        answer = history[-1][0].split('[ANSWER] ')[-1]

        qid_answer_map[question_id]['answers'].append(answer)
        qid_answer_map[question_id]['img_path'] = img_path
        qid_answer_map[question_id]['prompt'] = vlm_history[3]['input_text']


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
    if DATASET_NAME == 'OKVQA':
        qid_answer_map = defaultdict(lambda: defaultdict(list))
        for expdir in SAMPLE_EXPDIRS:
            read_okvqa_samples(expdir, qid_answer_map)

        # Evaluate
        sys.path.append('./src/evaluation')
        from vqaEval import VQAEval
        from vqa_tools import VQA
        question_path = "../vqa_data/KBVQA_data/ok-vqa/OpenEnded_mscoco_train2014_questions.json"
        annotation_path = "../vqa_data/KBVQA_data/ok-vqa/mscoco_train2014_annotations.json"

        vqa_helper = VQA(annotation_path, question_path)
        for idx in range(len(SAMPLE_EXPDIRS)):
            predictions = []
            for question_id, item in qid_answer_map.items():
                predictions.append({
                    'question_id': int(question_id),
                    'answer': item['answers'][idx]
                })

            vqaRes = vqa_helper.loadResFromDict(predictions)
            
            for question_id in tqdm(qid_answer_map, total=len(qid_answer_map), desc='calculating OKVQA score at example level...'):
                vqaEval = VQAEval(vqa_helper, vqaRes, n=2)
                vqaEval.evaluate(quesIds=[int(question_id)], verbose=False)
                score = vqaEval.accuracy['overall']
                qid_answer_map[question_id]['scores'].append(score)
                qid_answer_map[question_id]['gt_answer'] = vqaEval.vqa.qa[int(question_id)]['answers'][0]['answer']
                # print(qid_answer_map[question_id]['answers'])
                # print(qid_answer_map[question_id]['scores'])
        
        # flatten data
        flat_data = []
        for question_id, item in tqdm(qid_answer_map.items(), total=len(qid_answer_map), desc='flattening data...'):
            answers, scores, prompt, img_path, gt_answer = item['answers'], item['scores'], item['prompt'], item['img_path'], item['gt_answer']
            incorrect_answers = [ans for ans, s in zip(answers, scores) if s < 60.0]

            max_idx = np.argmax(scores)
            best_answer, best_score, mean_score = answers[max_idx], scores[max_idx], np.mean(scores)

            has_correct = True
            if best_score < 60.0:
                best_answer = gt_answer
                has_correct = False

            if len(incorrect_answers) == 0: # all correct; skip
                continue

            k = random.randint(0, len(incorrect_answers)-1)
            flat_data.append((question_id, img_path, best_score, mean_score, has_correct, gt_answer, best_answer, incorrect_answers[k], prompt))
        
        df = pd.DataFrame(flat_data, columns=['question_id', 'img_path', 'best_score', 'mean_score', 'has_correct', 'gt_answer', 'chosen', 'rejected', 'prompt'])
        """
        [2513 rows x 7 columns] # On these rows the model makes at least one error (out of 9k examples).
        Best overall score = 82.4
        Mean score = 51.7
        Has correct% = 84.3%  # Denominator = 2513
        """
    elif DATASET_NAME == 'EVQA':
        qid_answer_map = defaultdict(lambda: defaultdict(list))
        for expdir in SAMPLE_EXPDIRS:
            read_evqa_samples(expdir, qid_answer_map)

        # flatten data
        flat_data = []
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

            k = random.randint(0, len(incorrect_answers)-1)
            flat_data.append((question_id, img_path, best_score, mean_score, has_correct, gt_answer, best_answer, incorrect_answers[k], prompt))
        
        df = pd.DataFrame(flat_data, columns=['question_id', 'img_path', 'best_score', 'mean_score', 'has_correct', 'gt_answer', 'chosen', 'rejected', 'prompt'])
    elif DATASET_NAME == 'Infoseek' or DATASET_NAME == 'InfoseekNew':
        qid_answer_map = defaultdict(lambda: defaultdict(list))
        for expdir in SAMPLE_EXPDIRS:
            read_infoseek_samples(expdir, qid_answer_map)

        sys.path.append("./third_party/infoseek_eval")
        # if args.split in ['test', 'valid']:
            # reference_path = f"third_party/infoseek_eval/infoseek/infoseek_val.jsonl" #NOTE M2KR "test" split = official Infoseek "val" split
            # reference_qtype_path = f"third_party/infoseek_eval/infoseek/infoseek_val_qtype.jsonl"
        # elif args.split in ['train']:
        reference_path = f"third_party/infoseek_eval/infoseek/infoseek_train.jsonl" #NOTE M2KR "test" split = official Infoseek "val" split
        reference_qtype_path = None


        from infoseek_eval import evaluate_by_example, load_jsonl, prepare_qid2example
        reference = load_jsonl(reference_path)
        reference_qtype = load_jsonl(reference_qtype_path) if reference_qtype_path is not None else None
        qid2example = prepare_qid2example(reference, reference_qtype)
        # flatten data
        for idx in range(len(SAMPLE_EXPDIRS)):
            predictions = []
            for question_id, item in tqdm(qid_answer_map.items(), desc='evaluate example-by-example', total=len(qid_answer_map)):
                preds = [{
                    'data_id': question_id,
                    'prediction': item['answers'][idx]
                }]
                result, gt_answers = evaluate_by_example(preds, reference, reference_qtype, qid2example)
                score = max(result["unseen_entity_score"]["score"], result["final_score"], result["unseen_question_score"]["score"])
                qid_answer_map[question_id]['scores'].append(score)
                qid_answer_map[question_id]['gt_answer'] = gt_answers[0]
                # print(qid_answer_map[question_id]['answers'])
                # print(qid_answer_map[question_id]['gt_answer'])
                # print(qid_answer_map[question_id]['scores'])
                # breakpoint()

        flat_data = []
        for question_id, item in tqdm(qid_answer_map.items(), total=len(qid_answer_map), desc='flattening data...'):
            answers, scores, prompt, img_path, gt_answer = item['answers'], item['scores'], item['prompt'], item['img_path'], item['gt_answer']
            incorrect_answers = [ans for ans, s in zip(answers, scores) if s < 0.6]

            max_idx = np.argmax(scores)
            best_answer, best_score, mean_score = answers[max_idx], scores[max_idx], np.mean(scores)

            has_correct = True
            if best_score < 60.0:
                best_answer = gt_answer
                has_correct = False

            if len(incorrect_answers) == 0: # all correct; skip
                continue

            # print(answers, best_answer)
            # print(gt_answer)
            # print(best_score)
            # breakpoint()

            k = random.randint(0, len(incorrect_answers)-1)
            flat_data.append((question_id, img_path, best_score, mean_score, has_correct, gt_answer, best_answer, incorrect_answers[k], prompt))
        
        df = pd.DataFrame(flat_data, columns=['question_id', 'img_path', 'best_score', 'mean_score', 'has_correct', 'gt_answer', 'chosen', 'rejected', 'prompt'])
    else:
        raise NotImplementedError(f"Dataset name = {DATASET_NAME}")

    print(df)
    print(f"Best overall score = {df['best_score'].mean():.1f}")
    print(f"Mean score = {df['mean_score'].mean():.1f}")
    print(f"Has correct% = {df['has_correct'].mean() * 100:.1f}%")
    df.to_csv(TARGET_FILENAME)
    print(f"Pairwise data saved to {TARGET_FILENAME}")


