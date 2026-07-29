from datasets import load_dataset, load_from_disk , enable_progress_bar
from tqdm import tqdm
from typing import Any
from pathlib import Path
import json

def load_retrieval_jsonl(path: str, needed_qids: set) -> dict[str, dict[str, Any]]:
    by_qid: dict[str, dict[str, Any]] = {}
    with open(path) as f:
        for line in tqdm(f, desc="Reading retrieval JSONL"):
            record = json.loads(line)
            qid = record.get("question_id")
            if qid in needed_qids:
                by_qid[qid] = record
            else:
                continue
    # assert len(by_qid) == len(needed_qids), "Not all required QIDs are found!"
    return by_qid

"""
>>> ret_data['infoseek_val_00000000'].keys()
dict_keys(['question_id', 'image_id', 'question', 'answer', 'question_type', 'ground_truth_url', 'retrieved_pages'])

>>> ret_data['infoseek_val_00000000']['retrieved_pages'][:5]
[{'rank': 1, 'url': 'https://en.wikipedia.org/wiki/Volkswagen_EA827_engine', 'title': 'Volkswagen EA827 engine', 'score': 0.6004730463027954, 'knowledge_base_index': 71679}, {'rank': 2, 'url': 'https://en.wikipedia.org/wiki/Cruise_control', 'title': 'Cruise control', 'score': 0.5985638499259949, 'knowledge_base_index': 5380}, {'rank': 3, 'url': 'https://en.wikipedia.org/wiki/Nissan_SR_engine', 'title': 'Nissan SR engine', 'score': 0.5775291919708252, 'knowledge_base_index': 87308}, {'rank': 4, 'url': 'https://en.wikipedia.org/wiki/Nissan_MA_engine', 'title': 'Nissan MA engine', 'score': 0.5700928568840027, 'knowledge_base_index': 8148}, {'rank': 5, 'url': 'https://en.wikipedia.org/wiki/Chevrolet_Corvair', 'title': 'Chevrolet Corvair', 'score': 0.5683192014694214, 'knowledge_base_index': 9573}]

>>> target_ds
Dataset({
    features: ['pos_item_ids', 'pos_item_contents', 'img_id', 'img_path', 'image_id', 'question_id', 'question', 'answers', 'gold_answer', 'question_type', 'instruction', 'Pseudo Recall@1', 'Pseudo Recall@5', 'Pseudo Recall@10', 'Pseudo Recall@50', 'Pseudo Recall@100', 'Pseudo Recall@500', 'Recall@1', 'Recall@5', 'Recall@10', 'Recall@50', 'Recall@100', 'Recall@500', 'retrieved_passage', 'MRR'],
    num_rows: 3750
})

>>> target_ds[0]['retrieved_passage'][:5]
[{'passage_id': 'WikiWeb_Plumbago zeylanica_4', 'score': 45.09375}, {'passage_id': 'WikiWeb_Maclura pomifera_13', 'score': 44.34375}, {'passage_id': 'WikiWeb_Monotropa uniflora_6', 'score': 44.25}, {'passage_id': 'WikiWeb_Rumex confertus_2', 'score': 44.0}, {'passage_id': 'WikiWeb_Baccharis halimifolia_6', 'score': 43.90625}]

"""


url = (
    "hf://datasets/BByrneLab/"
    "multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR/"
    "Infoseek_data/test-00000-of-00001.parquet"
)

if __name__ == "__main__":
    enable_progress_bar()
    
    # m2kr_ds = load_dataset(
    #     "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", 
    #     "Infoseek_data", 
    #     split='test'
    # )
    
    m2kr_ds = load_dataset(
        "parquet",
        data_files={"test": url},
        split="test",
    )
    ds = load_dataset("Jingbiao/aravqa", data_dir="Infoseek_data", split="validation")
    
    idx = {qid: i for i, qid in enumerate(ds['question_id'])}
    ds = ds.select([idx[qid] for qid in m2kr_ds["question_id"]])
    

    needed_qids = set([str(qid) for qid in ds['question_id']])

    ret_data = load_retrieval_jsonl(
        "/workspace/projects/Infoseek-retrieval/outputs/infoseek_test_evaclip_top100.full.jsonl", 
        needed_qids=needed_qids
    )

    def add_retrieval_field(x):
        ret_info = ret_data.get(x['question_id'], None)
        if ret_info is None:
            retrieved_passage = None
        else:
            retrieved_passage = [
                {
                    'passage_id': page['url'],
                    'title': page['title'],
                    'score': page['score'],
                } for page in ret_info['retrieved_pages']
            ]
            pos_item_ids = [ret_info['ground_truth_url']]
            x['pos_item_ids'] = pos_item_ids
        
        x['retrieved_passage'] = retrieved_passage
        return x
    
    ds = ds.map(add_retrieval_field)
    ds = ds.filter(lambda x: x['retrieved_passage'] is not None)
    breakpoint()
    
    ds.save_to_disk("/workspace/projects/BERAG/outputs/jinghong_chen/Infoseek-testfull-with-retrieval")
    
        

    