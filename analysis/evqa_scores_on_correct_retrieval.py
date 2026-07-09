from datasets import load_from_disk
import pandas as pd


# ============================
SOURCE_DATASET_PATH = "outputs/jinghong_chen/EVQA-testfull-with-retrieval_post_reranked"
BASE_INFERENCE_PATH = "outputs/0925/EVQA/Qwen2-VL-2B-Instruct/marked_inference_results.csv"
# ============================

# RETRIEVAL_FIELD = "reranked_passage" # ["retrieved_passage", "reranked_passage"]
RETRIEVAL_FIELD = "retrieved_passage" # ["retrieved_passage", "reranked_passage"]
TOPK = 5


# ============================
# FIRST LINE = RETRIEVE RESULTS
# SECOND LINE = RERANK RESULTS

# MODEL_NAME = "2B-basemodel"
# MODEL_INFERENCE_PATH = "outputs/0925/EVQA/Qwen2-VL-2B-Instruct/marked_inference_results.csv"

# MODEL_NAME = "EVQA-SFT-1e-5"
# MODEL_INFERENCE_PATH = "outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-SFT-1e-5-0910/marked_inference_results.csv"
# MODEL_INFERENCE_PATH = "outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-SFT-Top5-Rerank-lr=1e-5-0910/marked_inference_results.csv"

# MODEL_NAME = "EVQA-AttnSFT[sum]-1e-5"
# MODEL_INFERENCE_PATH = "outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-AttnSFT-1e-5-0910/marked_inference_results.csv"
# MODEL_INFERENCE_PATH = "outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-AttnSFT-Top5-Rerank-lr=1e-5-0910/marked_inference_results.csv"


# MODEL_NAME = "EVQA-AttnSFT[max]-1e-5"
# MODEL_INFERENCE_PATH = "outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-AttnSFT-Agg=Max-Top5-Retrieve-lr=1e-5-0911/marked_inference_results.csv"
# MODEL_INFERENCE_PATH = "outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-AttnSFT-Agg=Max-Top5-Retrieve-lr=1e-5-0911/marked_inference_results.csv"

# MODEL_NAME = "EVQA-AttnSFT-QSpan-Agg=Sum-Top5-lr=1e-5"
# MODEL_INFERENCE_PATH = "outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-AttnSFT-QSpan-Agg=Sum-Top5-Retrieve-lr=1e-5-0916/marked_inference_results.csv"
# MODEL_INFERENCE_PATH = "outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-AttnSFT-QSpan-Agg=Sum-Top5-Rerank-lr=1e-5-0916/marked_inference_results.csv"

MODEL_NAME = "EVQA-AttnSFT-QSpan-Agg=LateInter-Top5-lr=1e-5"
MODEL_INFERENCE_PATH = "outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-AttnSFT-QSpan-Agg=LateInter-Top5-Retrieve-lr=1e-5-0916/marked_inference_results.csv"
# MODEL_INFERENCE_PATH = "outputs/0925/EVQA/Qwen2-VL-2B-Instruct-EVQA-AttnSFT-QSpan-Agg=LateInter-Top5-Rerank-lr=1e-5-0916/marked_inference_results.csv"
# ============================

if __name__ == "__main__":
    vqa_dataset = load_from_disk(SOURCE_DATASET_PATH)
    model_df = pd.read_csv(MODEL_INFERENCE_PATH)

    # create question_id - TopK hit dataframe (batched for speed)
    def _hit_on_topk_batch(batch):
        gt_doc_ids = [item[0] for item in batch['pos_item_ids']]
        retrieved_doc_ids_list = [
            {x['passage_id'] for x in retrieved[:TOPK]}
            for retrieved in batch[RETRIEVAL_FIELD]
        ]
        hits = [
            gt_doc_id in retrieved_doc_ids
            for gt_doc_id, retrieved_doc_ids in zip(gt_doc_ids, retrieved_doc_ids_list)
        ]
        return {"hit": hits}
    
    vqa_dataset = vqa_dataset.map(_hit_on_topk_batch, batched=True, batch_size=250, num_proc=4)

    dataset_df = vqa_dataset.to_pandas()
    dataset_df = dataset_df[['question_id', 'hit']]

    # join model_df and dataset_df on question_id
    model_df = model_df.merge(dataset_df, on='question_id', how='left')

    # report recall, scores, scores given hit, etc.
    recall = model_df['hit'].mean()

    score_given_hit = model_df[model_df['hit']]['score'].mean()

    # create question_id - base correct dataframe
    base_df = pd.read_csv(BASE_INFERENCE_PATH)
    base_df = base_df[['question_id', 'score']]
    base_df.rename(columns={'score': 'base_score'}, inplace=True)
    base_df['base_correct'] = (base_df['base_score'] >= 0.6)

    model_df = model_df.merge(base_df, on='question_id', how='left')

    # report score given wrong base and correct retrieval
    print("================================================")
    print(f"Results for model {MODEL_NAME}:")
    print(f"Recall: {recall}")
    print(f"Overall score: {model_df['score'].mean()}")
    print(f"Score given hit: {score_given_hit}")
    print(f"Score given miss: {model_df[~model_df['hit']]['score'].mean()}")
    print(f"Number of hit questions: {model_df['hit'].sum()}")
    print(f"Score given wrong base and correct retrieval: {model_df[~model_df['base_correct'] & model_df['hit']]['score'].mean()}")
    print(f"Number of wrong base and correct retrieval: {model_df[~model_df['base_correct'] & model_df['hit']].shape[0]} ({(model_df[~model_df['base_correct'] & model_df['hit']].shape[0] / model_df[model_df['hit']].shape[0]) * 100:.2f}%)")
    print("================================================")



