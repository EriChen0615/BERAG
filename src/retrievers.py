from abc import ABC, abstractmethod
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
)
from datasets import load_dataset, load_from_disk
from PIL import Image
from collections import defaultdict
import numpy as np
import torch
from vqa_datasets import load_passages
from tqdm import tqdm

class All_Retriever(ABC):
    @abstractmethod
    def query(self, query_text, query_img=None, question_id=0):
        pass

class SearchEnginerInterface_Retriever(All_Retriever):
    @abstractmethod
    def query(self, query_text, query_img=None, question_id=0):
        pass

    @abstractmethod
    def view_fulltext(self, query):
        pass
    
    @abstractmethod
    def next(self):
        pass



class DummyRetriever(All_Retriever):
    def __init__(self, always_return, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.always_return = always_return

    def query(self, query_text, query_img, question_id=0):
        return self.always_return

class OracleRetriever(All_Retriever):
    def __init__(self, ds_name, ds_subset, use_split, map_from_passage_set=False, passage_ds=None, passage_split=None):
        self.qid_to_gt_doc_map = {} #TODO
        self.ds = load_dataset(ds_name, ds_subset, split=use_split)
        if map_from_passage_set:
            self.passage_ds, self.pid_to_content_map = load_passages(passage_ds, split=passage_split)
            print(f"Using {passage_ds} as the passage set for OracleRetriever")
        for row in self.ds:
            self.qid_to_gt_doc_map[row['question_id']] = [
                {
                    "passage_id": row['pos_item_ids'][0],
                    "text": row['pos_item_contents'][0] if not map_from_passage_set else self.pid_to_content_map[row['pos_item_ids'][0]],
                    "score": 100.0,
                } 
            ]

    def query(self, query_text, query_img, ret_topk=1, top_k=1, question_id=0):
        gt_doc = self.qid_to_gt_doc_map[question_id]
        return gt_doc



class CacheDatasetRetriever(All_Retriever):
    def __init__(self, 
                ds_path,
                use_split,
                retrieval_field,
                passage_dataset_name,
                ret_topk,
                deduplicate=False,
        ):
        self.ret_topk = ret_topk
        self.deduplicate = deduplicate
        print(f"CacheDatasetRetriever: ret_topk={ret_topk}, deduplicate={deduplicate}")

        # import datasets
        # datasets.config.IN_MEMORY_MAX_SIZE = 85899345920 # 80GB
        self.ds_path = ds_path
        self.ds = load_from_disk(ds_path) # keep_in_memory doesn't speed up substantially
        self.retrieval_field = retrieval_field

        # from multiprocessing import Pool, cpu_count
        # from functools import partial
        # f = partial(process_row, retrieval_field=retrieval_field)
        
        # Use multiprocessing to create the map
        
        # with Pool(cpu_count() // 4) as pool:
            # results = list(tqdm(pool.imap(f, self.ds), total=len(self.ds), desc=f"Processing rows at {ds_path}"))
        # with ThreadPoolExecutor() as executor:
            # results = list(tqdm(executor.map(process_row, self.ds), desc=f"parsing retrieval dataset at {ds_path} into map...", total=len(self.ds)))
            # self.qid_to_ret_doc_map = dict(results)
        # self.qid_to_ret_doc_map = dict(results)


        # df = self.ds.to_pandas()
        # self.qid_to_ret_doc_map = df.set_index('question_id')[retrieval_field].to_dict()
        # self.qid_to_ret_doc_map = {}
        # for row in tqdm(self.ds, desc=f"parsing retrieval dataset at {ds_path} into map...", total=len(self.ds)):
            # self.qid_to_ret_doc_map[row['question_id']] = row[retrieval_field]
        # self.qid_to_ret_doc_map = {item['question_id']: item[retrieval_field] for item in tqdm(self.ds, desc=f"parsing retrieval dataset at {ds_path} into map...", total=len(self.ds))}
        self.qid_to_ret_doc_map = None
        # print(f"Using the {retrieval_field} field from the CachedDataset at {ds_path}")
        # breakpoint()
        passage_ds, self.pid_to_content_map = load_passages(passage_dataset_name, split=use_split)
    
    def query(self, query_text, query_img, top_k=1, question_id=0, ret_topk=1):
        if self.qid_to_ret_doc_map is None: #LAZY initialization
            self.qid_to_ret_doc_map = {item['question_id']: item[self.retrieval_field] for item in tqdm(self.ds, desc=f"parsing retrieval dataset at {self.ds_path} into map...", total=len(self.ds))}
            print(f"Using the {self.retrieval_field} field from the CachedDataset at {self.ds_path}")

        ret_docs = self.qid_to_ret_doc_map[question_id]
        for ret_doc in ret_docs:
            ret_doc['text'] = self.pid_to_content_map[ret_doc['passage_id']]

        if ret_topk is not None: # ret_topk has overriding priority.
            if self.deduplicate:
                seen_texts = set()
                deduped_docs = []
                for doc in ret_docs:
                    if doc['text'] not in seen_texts:
                        seen_texts.add(doc['text'])
                        deduped_docs.append(doc)
                    # else:
                        # print(f"Deduplicated doc: {doc['text']}")
                        # breakpoint()
                    if len(deduped_docs) == ret_topk:
                        break
                return deduped_docs[:ret_topk]
            return [doc for doc in ret_docs[:ret_topk]]
        else:
            if self.deduplicate:
                seen_texts = set()
                deduped_texts = []
                for doc in ret_docs:
                    if doc['text'] not in seen_texts:
                        seen_texts.add(doc['text'])
                        deduped_texts.append(doc['text'])
                    if len(deduped_texts) == self.ret_topk:
                        break
                return deduped_texts[:self.ret_topk]
            return [doc['text'] for doc in ret_docs[:self.ret_topk]]

        


from flmr import (
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
    FLMRModelForRetrieval, 
    create_searcher,
    search_custom_collection
)
class FLMRRetriever(All_Retriever):
    def __init__(
        self,
        ckpt_path,
        image_processor_name,
        searcher_kwargs,
        passage_ds,
        passage_subset,
        use_split,
        instruction,
        max_doc_len=2048, #NOTE not used!
        ret_topk=1,
        add_instruction=True,
        query_maxlen=32,
        use_gpu=False,
        *args, **kwargs
        ):
        self.ckpt_path = ckpt_path
        self.image_processor_name = image_processor_name
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(ckpt_path, query_maxlen=query_maxlen, subfolder="query_tokenizer")
        self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
            ckpt_path, subfolder="context_tokenizer"
        )
        self.flmr_model = None # Lazy initialization
        self.searcher_kwargs = searcher_kwargs
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_name)

        self.passage_ds = load_dataset(passage_ds, passage_subset, split=f'{use_split}_passages')
        print(f"FLMRRetriever using {passage_ds}/{passage_subset}. Split={use_split}")
        print("Size of passage set =", len(self.passage_ds))
        self.passage_contents = self.passage_ds["passage_content"]
        self.passage_ids = self.passage_ds["passage_id"]
        self.instruction = instruction
        self.max_doc_len = max_doc_len
        self.ret_topk = ret_topk
        self.add_instruction = add_instruction
        self.use_gpu = use_gpu
    
    def _init_model(self):
        self.flmr_model = FLMRModelForRetrieval.from_pretrained(
            self.ckpt_path,
            query_tokenizer=self.query_tokenizer,
            context_tokenizer=self.context_tokenizer,
        ).eval()
        if self.use_gpu:
            self.flmr_model = self.flmr_model.cuda()
            print("Using GPU for FLMR model")
        self.searcher = create_searcher(**self.searcher_kwargs)
    
    @torch.no_grad()
    def compute_query_embedding(self, query_text, query_img, no_image_search=False):
        tokenized_text = self.query_tokenizer([self.instruction + ":" + query_text if self.add_instruction else query_text])
        input_ids, attention_mask = tokenized_text['input_ids'], tokenized_text['attention_mask']
        if no_image_search:
            img = Image.new("RGB", (336, 336), color='black')
        else:
            img = Image.open(query_img)
        pixel_values = self.image_processor([img], return_tensors="pt").pixel_values

        query_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        query_embeddings = self.flmr_model.query(**query_input).late_interaction_output
        query_embeddings = query_embeddings.detach().cpu()

        return query_embeddings
    
    @torch.no_grad()
    def compute_query_embedding_batch(self, query_texts, query_imgs, no_image_search=False, use_doc_encoder_for_query=False):
        if use_doc_encoder_for_query:
            tokenized_texts = self.context_tokenizer([self.instruction + ":" + query_text if self.add_instruction else query_text for query_text in query_texts])
        else:
            tokenized_texts = self.query_tokenizer([self.instruction + ":" + query_text if self.add_instruction else query_text for query_text in query_texts])
        input_ids, attention_mask = tokenized_texts['input_ids'], tokenized_texts['attention_mask']
        if no_image_search:
            imgs = [Image.new("RGB", (336, 336), color='black') for query_img in query_imgs]
        else:
            imgs = [Image.open(query_img) for query_img in query_imgs]
        pixel_values = self.image_processor(imgs, return_tensors="pt").pixel_values

        query_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        if use_doc_encoder_for_query:
            query_embeddings = self.flmr_model.doc(**query_input).late_interaction_output
        else:
            query_embeddings = self.flmr_model.query(**query_input).late_interaction_output
        query_embeddings = query_embeddings.detach().cpu()

        return query_embeddings

    
    def query(self, query_text, query_img, topk=500, question_id=0, ret_topk=None): #NOTE topk=500 to replicate PreFLMR
        if self.flmr_model is None:
            self._init_model()
        custom_queries = {question_id: query_text}
        query_embeddings = self.compute_query_embedding(query_text, query_img)

        ranking = search_custom_collection(
            searcher=self.searcher,
            queries=custom_queries,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=topk, # how many documents to retrieve for each query
        )

        ranking_dict = ranking.todict()
        retrieved_docs = ranking_dict[question_id]
        retrieved_doc_scores = [doc[2] for doc in retrieved_docs]
        retrieved_docs = [doc[0] for doc in retrieved_docs]
        retrieved_doc_texts = [self.passage_contents[doc_idx] for doc_idx in retrieved_docs]
        retrieved_doc_ids = [self.passage_ids[doc_idx] for doc_idx in retrieved_docs]
        retrieved_doc_list = [
            {
                "passage_id": doc_id,
                "text": doc_text,
                "score": score,
            } for doc_id, score, doc_text in zip(retrieved_doc_ids, retrieved_doc_scores, retrieved_doc_texts)
        ]
        if ret_topk is not None: # ret_topk has overriding priority. 
            return [doc for doc in retrieved_doc_list[:ret_topk]]
        else: # kept for backward compatibility. but self.ret_topk should be abandoned
            if self.ret_topk == 1:
                return retrieved_doc_list[0]['text']
            else:
                return [doc['text'] for doc in retrieved_doc_list[:self.ret_topk]]
    
        # Query the whole dataset and return the recall 
    def query_and_evaluate_ds(
        self,
        ds,
        Ks=[1,5,10],
        topk=500,
        query_batch_size=8,
        compute_pseudo_recall=True,
        query_field='question',
        no_image_search=False,
        use_doc_encoder_for_query=False,
        save_retrieved_ds_to=None,
        return_ds=False,
    ): 
        if self.flmr_model is None:
            self._init_model()
        
        def compute_mrr(pos_ids, retrieved_doc_ids):
            """ Compute the Mean Reciprocal Rank (MRR) based on ground-truth pos_ids and retrieved doc ids """
            for rank, retrieved_doc_id in enumerate(retrieved_doc_ids, start=1):
                if retrieved_doc_id in pos_ids:
                    return 1 / rank
            return 0
        
        def encode_and_search_batch(batch, Ks=[1,5,10],topk=500, compute_pseudo_recall=True):
            custom_queries = {question_id: query_text for question_id, query_text in zip(batch["question_id"], batch[query_field])}
            query_embeddings = self.compute_query_embedding_batch(batch[query_field], batch["img_path"], no_image_search=no_image_search, use_doc_encoder_for_query=use_doc_encoder_for_query)

            # search
            ranking = search_custom_collection(
                searcher=self.searcher,
                queries=custom_queries,
                query_embeddings=query_embeddings,
                num_document_to_retrieve=topk, # how many documents to retrieve for each query
            )

            ranking_dict = ranking.todict()

            # Process ranking data and obtain recall scores
            # Psuedo Recall@K to be computed by matching the answer in the retrieved documents
            # Positive ids Recall@K to be computed by matching the sample positive id with the retrieved documents ids
            recall_dict = defaultdict(list)
            result_dict = defaultdict(list)
            mrr_list = []  # List to store MRR for each query

            for i, (question_id, pos_ids) in enumerate(zip(batch["question_id"], batch["pos_item_ids"])):
                retrieved_docs = ranking_dict[question_id]
                retrieved_doc_scores = [doc[2] for doc in retrieved_docs]
                retrieved_docs = [doc[0] for doc in retrieved_docs]
                retrieved_doc_texts = [self.passage_contents[doc_idx] for doc_idx in retrieved_docs]
                retrieved_doc_ids = [self.passage_ids[doc_idx] for doc_idx in retrieved_docs]
                retrieved_doc_list = [
                    {
                        "passage_id": doc_id,
                        "score": score,
                    } for doc_id, score in zip(retrieved_doc_ids, retrieved_doc_scores)
                ]
                result_dict["retrieved_passage"].append(retrieved_doc_list)

                # Compute MRR
                mrr = compute_mrr(pos_ids, retrieved_doc_ids)
                mrr_list.append(mrr)

                if compute_pseudo_recall:
                    # Psuedo Recall@K
                    hit_list = []
                    # Get answers
                    answers = batch["answers"][i]
                    for retrieved_doc_text in retrieved_doc_texts:
                        found = False
                        for answer in answers:
                            if answer.strip().lower() in retrieved_doc_text.lower():
                                found = True
                        if found:
                            hit_list.append(1)
                        else:
                            hit_list.append(0)

                    # print(hit_list)
                    # input()
                    for K in Ks:
                        recall = float(np.max(np.array(hit_list[:K])))
                        recall_dict[f"Pseudo Recall@{K}"].append(recall)
                
                # Positive ids Recall@K    
                retrieved_doc_ids = [self.passage_ids[doc_idx] for doc_idx in retrieved_docs] 
                hit_list = []
                for retrieved_doc_id in retrieved_doc_ids:
                    found = False
                    for pos_id in pos_ids:
                        if pos_id == retrieved_doc_id:
                            found = True
                    if found:
                        hit_list.append(1)
                    else:
                        hit_list.append(0)
                for K in Ks:
                    recall = float(np.max(np.array(hit_list[:K])))
                    recall_dict[f"Recall@{K}"].append(recall)
            batch.update(recall_dict)
            batch.update(result_dict)
            batch["MRR"] = mrr_list  # Add MRR to the batch
            return batch

        ds = ds.map(
            encode_and_search_batch,
            fn_kwargs={"Ks": Ks, "topk": topk, "compute_pseudo_recall": compute_pseudo_recall},
            batched=True,
            batch_size=query_batch_size,
            load_from_cache_file=False,
            new_fingerprint="avoid_cache",
        )

        if save_retrieved_ds_to is not None:
            ds.save_to_disk(save_retrieved_ds_to)
            print("Retrieval Dataset saved to", save_retrieved_ds_to)

        dict_to_report = {}
        
        if compute_pseudo_recall:
            for K in Ks:
                recall = np.mean(np.array(ds[f"Pseudo Recall@{K}"]))
                print(f"Pseudo Recall@{K}:\t", recall)
                dict_to_report[f"Pseudo Recall@{K}"] = recall
        for K in Ks:
            recall = np.mean(np.array(ds[f"Recall@{K}"]))
            print(f"Recall@{K}:\t", recall)
            dict_to_report[f"Recall@{K}"] = recall
        
        # Convert dataset to pandas DataFrame
        df = ds.to_pandas()

        # Aggregate by question_id, finding the best MRR
        aggregated_df = df.groupby('question_id').apply(lambda group: group.loc[group['MRR'].idxmax()])

        # Report the best Recall@1, Recall@5, ..., for each question_id
        dict_to_report[f"MRR"] = df['MRR'].mean()
        dict_to_report[f"Best MRR"] = aggregated_df['MRR'].mean()
        print(f"MRR = {df['MRR'].mean()}")
        print(f"Best MRR = {aggregated_df['MRR'].mean()}")

        for K in Ks:
            best_recall_at_K = aggregated_df[f"Recall@{K}"].mean()
            print(f"Best Recall@{K}:\t", best_recall_at_K)
            dict_to_report[f"Best Recall@{K}"] = best_recall_at_K

        if compute_pseudo_recall:
            for K in Ks:
                best_pseudo_recall_at_K = aggregated_df[f"Pseudo Recall@{K}"].mean()
                print(f"Best Pseudo Recall@{K}:\t", best_pseudo_recall_at_K)
                dict_to_report[f"Best Pseudo Recall@{K}"] =  best_pseudo_recall_at_K

        print("=============================")
        if not return_ds:
            return dict_to_report
        else:
            return ds, dict_to_report
    
        
    

class SE_FLMRRetriever(FLMRRetriever):
    def __init__(self, 
                 num_doc_to_return=5,
                 preview_max_wordcount=20,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_retrieved_docs = []
        self.cache_idx = 0
        self.num_doc_to_return = num_doc_to_return
        self.preview_max_wordcount = preview_max_wordcount

    def query(self, query_text, query_img, topk=100, question_id=0):
        self.cache_idx = 0
        if self.flmr_model is None:
            self._init_model()
        custom_queries = {question_id: query_text}
        query_embeddings = self.compute_query_embedding(query_text, query_img)

        ranking = search_custom_collection(
            searcher=self.searcher,
            queries=custom_queries,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=topk, # how many documents to retrieve for each query
        )

        ranking_dict = ranking.todict()
        retrieved_docs = ranking_dict[question_id]
        retrieved_doc_scores = [doc[2] for doc in retrieved_docs]
        retrieved_docs = [doc[0] for doc in retrieved_docs]
        retrieved_doc_texts = [self.passage_contents[doc_idx] for doc_idx in retrieved_docs]
        retrieved_doc_ids = [self.passage_ids[doc_idx] for doc_idx in retrieved_docs]
        retrieved_doc_list = [
            {
                "idx": idx,
                "title": doc_id.split('_')[1],
                "passage_id": doc_id,
                "text": doc_text,
                "score": score,
            } for idx, (doc_id, score, doc_text) in enumerate(zip(retrieved_doc_ids, retrieved_doc_scores, retrieved_doc_texts))
        ]
        self.cache_retrieved_doc = retrieved_doc_list
        self.cache_idx = self.num_doc_to_return
        previews_to_return = [
            {
                "idx": doc["idx"],
                "title": doc["title"],
                "passage_id": doc['passage_id'],
                "text": self._make_preview(doc['text']),
                "score": doc['score'],
            } for doc in self.retrieved_doc_list[:self.cache_idx]
        ]
        return previews_to_return
    
    def _make_preview(self, doc_text):
        space_delimited_words = doc_text.split(' ')
        preview_words = space_delimited_words[:self.preview_max_wordcount]
        return " ".join(preview_words)

    def view_fulltext(k):
        k = int(k)
        if k < len(self.cache_retrieved_docs):
            return self.cache_retrieved_docs[k]['text'][:self.max_doc_len]
        else:
            return [f"RETRIEVER ERROR! Document with index {k} is not available!"]
    
    def next():
        self.cache_idx += 5
        if self.cache_idx < len(self.cache_retrieved_doc):
            return self.cache_retrieved_doc[self.cache_idx-5:self.cache_idx]
        else:
            return [f"RETRIEVER ERROR! Document with index {self.cache_idx} is not available!"]

    
