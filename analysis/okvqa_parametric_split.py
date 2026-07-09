NAME_TO_EXPDIRS_TO_EVAL_MAP = {
    "2B_RAG-SFT": "outputs/20241213-15-OKVQA_valid-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-2B_rag1-ft_ckpt-1686_PreFLMR-L",
    # "7B_RAG-SFT": "outputs/20241213-17-OKVQA_valid-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=3_QWen2VL-7B_rag1-ft_ckpt-1686_PreFLMR-L",
    "7B_RAG-SFT": "outputs/20241213-15-OKVQA_valid-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag1-ft_ckpt-1686_PreFLMR-L",
    "2B_NoRAG-SFT": "outputs/20241210-13-OKVQA_valid-0_NoRAGRead_QWen2VL-2B",
    "7B_NoRAG-SFT": "outputs/20241211-11-OKVQA_valid-0_NoRAGRead_QWen2VL-7B-norag-ckpt843",
    # "7B_RAG-SFT+DPO": "",
    "2B_RAG-SFT+SFT": "outputs/20241216-12-OKVQA_valid-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-2B_rag5-ft_ckpt-234_PreFLMR-L",
    # "2B_RAG-SFT+DPO": "outputs/20241217-15-OKVQA_valid-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-2B_rag5-dpo_beta=0.1_ckpt-234_PreFLMR-L",
    "2B_RAG-SFT+DPO": "outputs/20250120-12-OKVQA_valid-0_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-2B_rag5-dpo_beta=0.1_ckpt-78_PreFLMR-L",
}

# PARAMETRIC_LABEL_FROM="7B_NoRAG-SFT"
PARAMETRIC_LABEL_FROM="2B_NoRAG-SFT"

import sys
import os
import json
from tqdm import tqdm
import pandas as pd
sys.path.append('./src/evaluation')
from vqaEval import VQAEval
from vqa_tools import VQA

def read_and_mark_okvqa_results_as_df(expdir):
    # setup 
    question_path = "../vqa_data/KBVQA_data/ok-vqa/OpenEnded_mscoco_val2014_questions.json"
    annotation_path = "../vqa_data/KBVQA_data/ok-vqa/mscoco_val2014_annotations.json"
    vqa_helper = VQA(annotation_path, question_path)

    with open(os.path.join(expdir, "histories.json")) as f:
        histories = json.load(f)

    predictions = []
    for history in tqdm(histories, total=len(histories), desc=f"Reading from {expdir}"):
        question_id = history[0][0]['question_id']
        answer = history[-1][0].split('[ANSWER] ')[-1]
        predictions.append({
            'question_id': int(question_id),
            'answer': answer
        })

    vqaRes = vqa_helper.loadResFromDict(predictions)
    vqaEval = VQAEval(vqa_helper, vqaRes, n=2)

    data = []
    for history in tqdm(histories, total=len(histories), desc=f"Reading from {expdir}"):
        question_id = history[0][0]['question_id']
        question = history[0][0]['question']
        answer = history[-1][0].split('[ANSWER] ')[-1]
        img_path = history[0][0]['img_path']
        vqaEval.evaluate(quesIds=[int(question_id)], verbose=False)
        score = vqaEval.accuracy['overall']
        gt_answer = vqaEval.vqa.qa[int(question_id)]['answers'][0]['answer']
        data.append((question_id, img_path, question, gt_answer, answer, score))
        # print(data[-1])
        # breakpoint()
    
    return pd.DataFrame(data, columns=['question_id', 'img_path', 'question', 'gt_answer', 'answer', 'score'])

if __name__ == "__main__":
    # Read in qid_answer_map from each experiment
    df_map = {}
    for name, expdir in NAME_TO_EXPDIRS_TO_EVAL_MAP.items():
        df_map[name] = read_and_mark_okvqa_results_as_df(expdir)
    
    # Label those answered correctly in the `PARAMETRIC_LABEL_FROM` df as `parametric_ok`
    parametric_df = df_map[PARAMETRIC_LABEL_FROM]
    parametric_ok = parametric_df['score'] >= 60.0

    parametric_len = parametric_ok.sum()
    external_len = len(parametric_df) - parametric_len

    for name in df_map:
        df_map[name]['parametric_ok'] = parametric_ok
        df = df_map[name]
        print(f"System name = {name}")
        print(f"Overall score = {df['score'].mean()}")
        print(f"Parametric Split ({parametric_len}) score = {df[df['parametric_ok'] == True]['score'].mean()}")
        print(f"External Split ({external_len}) score = {df[df['parametric_ok'] == False]['score'].mean()}")
        print("="*50)
    breakpoint()

    # Look at some cases where RAG answers correctly but NoRAG answers incorrectly
    norag_df = df_map['7B_NoRAG-SFT']
    rag_df = df_map['7B_RAG-SFT']

    rag_correct = rag_df['score'] >= 60.0
    norag_correct = norag_df['score'] >= 60.0

    both_correct = (rag_correct & norag_correct).sum()
    both_wrong = (~rag_correct & ~norag_correct).sum()
    rag_correct_and_norag_wrong =  (rag_correct & ~norag_correct).sum()
    rag_wrong_and_norag_correct =  (~rag_correct & norag_correct).sum()
    print("Both Correct:", both_correct)
    print("RAG correct & NoRAG wrong", rag_correct_and_norag_wrong)
    print("RAG wrong & NoRAG correct", rag_wrong_and_norag_correct)
    print("Both Wrong:", both_wrong)
    breakpoint()
    pass

    # Qualitative Cases where both are wrong
    both_wrong_idx = (~rag_correct & ~norag_correct)
    both_wrong_rag = rag_df[both_wrong_idx]
    both_wrong_norag = norag_df[both_wrong_idx]
    for rag_item, norag_item in zip(both_wrong_rag.itertuples(), both_wrong_norag.itertuples()):
        break 
        print("Both Wrong")
        print(f"Image path: {rag_item.img_path}")
        print(f"Question: {rag_item.question}")
        print(f"GT Answer: {rag_item.gt_answer}")
        print(f"RAG system answer: {rag_item.answer}")
        print(f"NoRAG system answer: {norag_item.answer}")
        breakpoint()
    

    norag_wrong_idx = (~norag_correct) 
    df1, df2 = rag_df[norag_wrong_idx], norag_df[norag_wrong_idx]
    for rag_item, norag_item in zip(df1.itertuples(), df2.itertuples()):
        print("NoRAG wrong")
        print(f"Image path: {rag_item.img_path}")
        print(f"Question ID: {rag_item.question_id}")
        print(f"Question: {rag_item.question}")
        print(f"GT Answer: {rag_item.gt_answer}")
        print(f"RAG system answer: {rag_item.answer}")
        print(f"NoRAG system answer: {norag_item.answer}")
        breakpoint()
        
        
    rag_correct_norag_wrong_idx = (rag_correct & ~norag_correct) 
    df1, df2 = rag_df[rag_correct_norag_wrong_idx], norag_df[rag_correct_norag_wrong_idx]
    for rag_item, norag_item in zip(df1.itertuples(), df2.itertuples()):
        print("RAG correct & NoRAG wrong")
        print(f"Image path: {rag_item.img_path}")
        print(f"Question ID: {rag_item.question_id}")
        print(f"Question: {rag_item.question}")
        print(f"GT Answer: {rag_item.gt_answer}")
        print(f"RAG system answer: {rag_item.answer}")
        print(f"NoRAG system answer: {norag_item.answer}")
        breakpoint()


    
        
