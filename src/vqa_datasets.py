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
        Infoseek_ds = load_dataset("Jingbiao/aravqa", "Infoseek_data", split=split)
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

def load_passages(dataset_name, split='test'):
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
        Infoseek_passage_ds = load_dataset("Jingbiao/aravqa", "Infoseek_passages", split=f"{split}_passages")
        print(f"Using Jingbiao/aravqa InfoseekNew. Split={split}")
        print("Size of passage set = ", len(Infoseek_passage_ds))
        pid_to_content_map = {item['passage_id']: item['passage_content'] for item in Infoseek_passage_ds}
        return Infoseek_passage_ds, pid_to_content_map
    elif dataset_name == 'InfoseekNew_FullPassage':
        requested_split = f"{split}_passages"
        try:
            Infoseek_passage_ds = load_dataset("Jingbiao/aravqa", "InfoseekFull_passages", split=requested_split)
            resolved_split = requested_split
        except ValueError as e:
            # Some cached variants expose only train/valid passage splits.
            if split == "test" and "Unknown split" in str(e):
                fallback_split = "valid_passages"
                print(
                    f"[load_passages] Requested split '{requested_split}' not available for "
                    f"InfoseekFull_passages; falling back to '{fallback_split}'."
                )
                Infoseek_passage_ds = load_dataset("Jingbiao/aravqa", "InfoseekFull_passages", split=fallback_split)
                resolved_split = fallback_split
            else:
                raise
        print(f"Using Jingbiao/aravqa Infoseek Full Passages. Split={resolved_split}")
        print("Size of passage set = ", len(Infoseek_passage_ds))
        pid_to_content_map = {item['passage_id']: item['passage_content'] for item in Infoseek_passage_ds}
        return Infoseek_passage_ds, pid_to_content_map
    else:
        raise NotImplementedError("load_vqa_passages")
