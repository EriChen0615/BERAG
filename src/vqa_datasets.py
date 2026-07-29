import json
import os
from datasets import load_dataset, load_from_disk

def load_vqa_dataset(dataset_name, split='test', img_basedir='data/', take_n=-1, seed=0):
    def add_prefix(row):
        row['img_path'] = f"{img_basedir}/{row['img_path']}"
        return row

    if dataset_name == 'EVQA':
        EVQA_ds = load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", "EVQA_data", split=split)
        EVQA_ds = EVQA_ds.map(add_prefix)
        if take_n > 0:
            import random
            random.seed(seed)
            EVQA_ds = EVQA_ds.select(random.sample(range(len(EVQA_ds)), k=take_n))
        return EVQA_ds
    elif dataset_name == 'EVQA_with_evidence':
        EVQA_ds = load_from_disk("outputs/0jingbiao_mei/EVQA-testfull-with-retrieval-rerank7B-step4000_post_reranked")
        EVQA_ds = EVQA_ds.map(add_prefix)
        if take_n > 0:
            import random
            random.seed(seed)
            EVQA_ds = EVQA_ds.select(random.sample(range(len(EVQA_ds)), k=take_n))
        return EVQA_ds
    elif dataset_name == 'OKVQA':
        OKVQA_ds = load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", "OKVQA_data", split=split)
        OKVQA_ds = OKVQA_ds.map(add_prefix)
        if take_n > 0:
            import random
            random.seed(seed)
            OKVQA_ds = OKVQA_ds.select(random.sample(range(len(OKVQA_ds)), k=take_n))
        return OKVQA_ds
    elif dataset_name == 'OKVQA-heldout':
        OKVQA_ds = load_from_disk("data/jinghong_chen/OKVQA-heldout")
        OKVQA_ds = OKVQA_ds.map(add_prefix)
        if take_n > 0:
            import random
            random.seed(seed)
            OKVQA_ds = OKVQA_ds.select(random.sample(range(len(OKVQA_ds)), k=take_n))
        return OKVQA_ds
    elif dataset_name == 'Infoseek':
        Infoseek_ds = load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", "Infoseek_data", split=split)
        Infoseek_ds = Infoseek_ds.map(add_prefix)
        if take_n > 0:
            import random
            random.seed(seed)
            Infoseek_ds = Infoseek_ds.select(random.sample(range(len(Infoseek_ds)), k=take_n))
        return Infoseek_ds
    elif dataset_name == 'InfoseekNew':
        Infoseek_ds = load_dataset("Jingbiao/aravqa", data_dir="Infoseek_data", split=split)
        Infoseek_ds = Infoseek_ds.map(add_prefix)
        if take_n > 0:
            import random
            random.seed(seed)
            Infoseek_ds = Infoseek_ds.select(random.sample(range(len(Infoseek_ds)), k=take_n))
        return Infoseek_ds
    elif dataset_name == 'InfoseekNew_with_evidence':
        Infoseek_ds = load_from_disk("outputs/0jingbiao_mei/InfoseekNew-test_full-with-retrieval-CLS7B_post_reranked")
        Infoseek_ds = Infoseek_ds.map(add_prefix)
        if take_n > 0:
            import random
            random.seed(seed)
            Infoseek_ds = Infoseek_ds.select(random.sample(range(len(Infoseek_ds)), k=take_n))
        return Infoseek_ds
    elif dataset_name == 'EVQA2hop':
        EVQA2hop_ds = load_dataset("Jingbiao/aravqa", "EVQA2hop_1013_data", split=split)
        EVQA2hop_ds = EVQA2hop_ds.map(add_prefix)
        return EVQA2hop_ds
    elif dataset_name == 'dummy':
        return  [
            {'question': "Is this plant poisonous?",
             'img_path': "data/test_data/EVQA0.PNG", 
             'gold_answer': "yes"}
        ]
    else:
        raise NotImplementedError(f"{dataset_name} not implemented!")

def _reconstruct_echosight_article(entry):
    title = str(entry.get('title') or '')
    section_titles = entry.get('section_titles') or []
    section_texts = entry.get('section_texts') or []
    pieces = []
    if title:
        pieces.append(title)
    for sec_title, sec_text in zip(section_titles, section_texts):
        sec_title = str(sec_title or '').strip()
        sec_text = str(sec_text or '').strip()
        if sec_title:
            pieces.append(sec_title)
        if sec_text:
            pieces.append(sec_text)
    if not pieces:
        pieces.append(json.dumps(entry, ensure_ascii=False))
    return '\n'.join(pieces)


def _load_echosight_kb_passages(kb_path):
    with open(kb_path) as f:
        kb_by_url = json.load(f)
    passage_ids = []
    titles = []
    sources = []
    knowledge_base_indices = []
    pid_to_content_map = {}
    for kb_index, entry in enumerate(kb_by_url.values()):
        passage_id = entry['url']
        passage_ids.append(passage_id)
        titles.append(entry.get('title') or passage_id)
        sources.append('echosight_kb')
        knowledge_base_indices.append(kb_index)
        pid_to_content_map[passage_id] = _reconstruct_echosight_article(entry)

    # Keep full document text out of the Arrow Dataset. ragk_answer_ppl.py only
    # needs passage_id/title from passage_ds; full text lives in pid_to_content_map.
    from datasets import Dataset
    passage_ds = Dataset.from_dict({
        'passage_id': passage_ids,
        'title': titles,
        'source': sources,
        'knowledge_base_index': knowledge_base_indices,
    })
    print(f"Using EchoSight KB passages from {kb_path}.")
    print("Size of passage set = ", len(passage_ds))
    return passage_ds, pid_to_content_map


def _resolve_jingbiao_aravqa_split(split):
    if split in {'train', 'train_passages'}:
        return 'train'
    if split in {'valid', 'validation', 'val', 'valid_passages', 'validation_passages', 'test', 'test_passages'}:
        return 'validation'
    return split.removesuffix('_passages')


def load_passages(dataset_name, split='test'):
    dataset_name_str = str(dataset_name)
    if dataset_name_str.startswith('EchoSightKB:'):
        return _load_echosight_kb_passages(dataset_name_str.split(':', 1)[1])
    if dataset_name == 'EchoSightKB':
        kb_path = os.environ.get('ECHOSIGHT_KB_PATH')
        if not kb_path:
            raise ValueError("Set ECHOSIGHT_KB_PATH or pass --passage_set_name EchoSightKB:/path/to/wiki_100_dict_v4.json")
        return _load_echosight_kb_passages(kb_path)
    if os.path.exists(str(dataset_name)):
        passage_ds = load_from_disk(str(dataset_name))
        print(f"Using local passage dataset from {dataset_name}.")
        print("Size of passage set = ", len(passage_ds))
        pid_to_content_map = {item['passage_id']: item['passage_content'] for item in passage_ds}
        return passage_ds, pid_to_content_map
    if dataset_name == 'EVQA': 
        EVQA_PASSAGE_DS = load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", "EVQA_passages", split=f"{split}_passages")
        print(f"Using BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR EVQA. Split={split}")
        print("Size of passage set = ", len(EVQA_PASSAGE_DS))
        # pid_to_content_map = {item['passage_id']: item['passage_content'] for item in EVQA_PASSAGE_DS}
        pid_to_content_map = dict(
            zip(
                EVQA_PASSAGE_DS["passage_id"],
                EVQA_PASSAGE_DS["passage_content"],
            )
        )
        return EVQA_PASSAGE_DS, pid_to_content_map
    elif dataset_name == 'OKVQA':
        OKVQA_PASSAGE_DS = load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", "OKVQA_passages", split=f"{split}_passages")
        print(f"Using BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR OKVQA. Split={split}")
        print("Size of passage set = ", len(OKVQA_PASSAGE_DS))
        pid_to_content_map = {item['passage_id']: item['passage_content'] for item in OKVQA_PASSAGE_DS}
        return OKVQA_PASSAGE_DS, pid_to_content_map
    elif dataset_name == 'Infoseek':
        Infoseek_passage_ds = load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", "Infoseek_passages", split=f"{split}_passages")
        print(f"Using BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR Infoseek. Split={split}")
        print("Size of passage set = ", len(Infoseek_passage_ds))
        pid_to_content_map = {item['passage_id']: item['passage_content'] for item in Infoseek_passage_ds}
        return Infoseek_passage_ds, pid_to_content_map
    elif dataset_name == 'InfoseekNew':
        resolved_split = _resolve_jingbiao_aravqa_split(split)
        Infoseek_passage_ds = load_dataset("Jingbiao/aravqa", data_dir="Infoseek_passages", split=resolved_split)
        print(f"Using Jingbiao/aravqa InfoseekNew. Split={resolved_split}")
        print("Size of passage set = ", len(Infoseek_passage_ds))
        pid_to_content_map = {item['passage_id']: item['passage_content'] for item in Infoseek_passage_ds}
        return Infoseek_passage_ds, pid_to_content_map
    elif dataset_name == 'InfoseekNew_FullPassage':
        resolved_split = _resolve_jingbiao_aravqa_split(split)
        Infoseek_passage_ds = load_dataset("Jingbiao/aravqa", data_dir="InfoseekFull_passages", split=resolved_split)
        print(f"Using Jingbiao/aravqa Infoseek Full Passages. Split={resolved_split}")
        print("Size of passage set = ", len(Infoseek_passage_ds))
        pid_to_content_map = {item['passage_id']: item['passage_content'] for item in Infoseek_passage_ds}
        return Infoseek_passage_ds, pid_to_content_map
    else:
        raise NotImplementedError("load_vqa_passages")
