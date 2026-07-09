# EXPDIR="outputs/20250121-15-InfoseekNew_valid_m2kr-256_OracleRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_rag1-ft_ckpt-2000_PreFLMR-L"
# EXPDIR="outputs/20250122-15-InfoseekNew_valid_m2kr-256_OracleRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_pretrained_PreFLMR-L"
NAME_TO_EXPDIRS_TO_EVAL_MAP = {
    # "2B_pretrained": "",
    # "7B_pretrained", ""
    # "2B_RAG-SFT": "",
    # "2B_NoRAG-SFT": "",
    # "7B_NoRAG-SFT": "",
    # "2B_RAG-SFT+SFT": "",
    # "2B_RAG-SFT+DPO": "",
    # "7B_PreFLMR_RAG-SFT": "",
    # "7B_GoldRetrieval_RAG-SFT": "outputs/20250121-15-InfoseekNew_valid_m2kr-256_OracleRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_rag1-ft_ckpt-2000_PreFLMR-L",
    # "7B_PseudoGoldRetrieval_RAG-SFT": "outputs/20250121-16-Infoseek_test-256_OracleRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_rag1-ft_ckpt-2000_PreFLMR-L",
    "NoRetrieval_2B-pretrained": "outputs/20241209-17-Infoseek_test-256_NoRAGRead_QWen2VL-2B",
    # "NoRetrieval_7B-pretrained": "",
    # "NoRetrieval_2B-direct-SFT": "",
    "NoRetrieval_7B-direct-SFT": "outputs/20241212-14-Infoseek_test-256_NoRAGRead_QWen2VL-7B_ckpt2000",

    # "Retrieval-K=1_7B-pretrained": "",
    "Retrieval-K=1_7B-RAG1-SFT": "outputs/20241213-14-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L",
    "Retrieval-K=5_7B-RAG1-SFT": "outputs/20241213-14-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L",
    "Retrieval-K=1_7B-RAG5-SFT": "outputs/20250121-14-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_rag5-sft-ckpt718_PreFLMR-L",
    "Retrieval-K=5_7B-RAG5-SFT": "outputs/20250121-14-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=5_QWen2VL-7B-LoRA_rag5-sft-ckpt718_PreFLMR-L",

    # "RetrievalReranking-K=1_7B-pretrained": "",
    "RetrievalReranking-K=1_7B-RAG1-SFT": "outputs/20241213-15-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L",
    "RetrievalReranking-K=5_7B-RAG1-SFT": "outputs/20241213-15-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L",
    "RetrievalReranking-K=1_7B-RAG5-SFT": "outputs/20241229-03-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag5-sft_PreFLMR-L",
    "RetrievalReranking-K=5_7B-RAG5-SFT": "outputs/20241229-03-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag5-sft_PreFLMR-L",
}

# EXPDIR="outputs/20241212-14-Infoseek_test-256_NoRAGRead_QWen2VL-7B_ckpt2000"
# EXPDIR="outputs/20241213-14-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L"
EXPDIRS=[
    # "outputs/20241213-14-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L",
    # "outputs/20250121-14-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_rag5-sft-ckpt718_PreFLMR-L",
    # "outputs/20250121-14-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=5_QWen2VL-7B-LoRA_rag5-sft-ckpt718_PreFLMR-L",
    # "outputs/20250122-14-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag5-dpo-beta=1.0-ckpt718_PreFLMR-L",
    # "outputs/20250122-14-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=10_QWen2VL-7B_rag5-dpo-beta=1.0-ckpt718_PreFLMR-L",
    # "outputs/20241213-15-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L",
    # "outputs/20241213-15-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L",
    # "outputs/20241213-15-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=10_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L",
    # "outputs/2,0250122-17-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_pretrained_PreFLMR-L",
    # "outputs/20250122-17-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=5_QWen2VL-7B-LoRA_pretrained_PreFLMR-L",
    # "outputs/20250122-18-Infoseek_test-256_CacheRetrieve[TopK]-Read_RetrieveTopK=10_QWen2VL-7B-LoRA_pretrained_PreFLMR-L",
    # "outputs/20250122-18-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_pretrained_PreFLMR-L",
    # "outputs/20250122-18-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_pretrained_PreFLMR-L",
    # "outputs/20250122-18-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=10_QWen2VL-7B_pretrained_PreFLMR-L",
    # "outputs/20250122-18-Infoseek_test-256_NoRAGRead_QWen2VL-7B_pretrained",
    # "outputs/20250122-18-Infoseek_test-256_NoRAGRead_QWen2VL-7B_pretrained",
    # "outputs/20250123-11-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag5-dpo-nosft-beta=1.2-ckpt718_PreFLMR-L",
    # "outputs/20250123-11-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=3_QWen2VL-7B_rag5-dpo-nosft-beta=1.2-ckpt718_PreFLMR-L",
    # "outputs/20250123-11-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag5-dpo-nosft-beta=1.2-ckpt718_PreFLMR-L",
    # "outputs/20250123-11-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=10_QWen2VL-7B_rag5-dpo-nosft-beta=1.2-ckpt718_PreFLMR-L",
    # "outputs/20250123-11-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag5-dpo-nosft-beta=1.0-ckpt718_PreFLMR-L",
    # "outputs/20250123-11-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=3_QWen2VL-7B_rag5-dpo-nosft-beta=1.0-ckpt718_PreFLMR-L",
    # "outputs/20250123-11-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag5-dpo-nosft-beta=1.0-ckpt718_PreFLMR-L",
    # "outputs/20250123-11-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=10_QWen2VL-7B_rag5-dpo-nosft-beta=1.0-ckpt718_PreFLMR-L",
    # "outputs/20250123-13-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag5-dpo-nosft-beta=1.5-ckpt718_PreFLMR-L",
    # "outputs/20250123-13-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=3_QWen2VL-7B_rag5-dpo-nosft-beta=1.5-ckpt718_PreFLMR-L",
    # "outputs/20250123-13-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag5-dpo-nosft-beta=1.5-ckpt718_PreFLMR-L",
    # "outputs/20250123-13-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=10_QWen2VL-7B_rag5-dpo-nosft-beta=1.5-ckpt718_PreFLMR-L",
    # "outputs/20250124-13-InfoseekNew_valid_m2kr-256_FullPassageOracleRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_pretrained_PreFLMR-L",
    # "outputs/20250124-14-InfoseekNew_valid_m2kr-256_FullPassageOracleRetrieve[TopK]-Read_RetrieveTopK=1_GPT4o-mini_pretrained_PreFLMR-L"
    # "outputs/20250124-17-InfoseekNew_valid_m2kr-256_OracleRetrieve[TopK]-Read_RetrieveTopK=1_GPT4o-mini_pretrained_PreFLMR-L",
    # "outputs/20250123-11-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=10_QWen2VL-7B_rag5-dpo-nosft-beta=1.0-ckpt718_PreFLMR-L",
    # "outputs/20241229-03-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag5-sft_PreFLMR-L",
    # "outputs/20241229-03-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag5-sft_PreFLMR-L",
    # "outputs/20250123-11-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag5-dpo-nosft-beta=0.1-ckpt500_PreFLMR-L",
    # "outputs/20250124-18-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag5-dpo-nosft-beta=0.1-ckpt500_PreFLMR-L",
    # "outputs/20250124-18-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=10_QWen2VL-7B_rag5-dpo-nosft-beta=0.1-ckpt500_PreFLMR-L",
    "outputs/20250124-17-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=5_QWen2VL-7B_rag5-dpo-nosft-beta=0.1-ckpt500_PreFLMR-L",
]
IMG_BASEDIR = ''
OPENAI_MODEL='gpt-4o-mini-2024-07-18'
# EXPDIR="outputs/20241213-15-Infoseek_test-256_CacheRetrieveRerank[TopK]-Read_RetrieveTopK=1_QWen2VL-7B_rag1-ft_ckpt-2000_PreFLMR-L"


from collections import defaultdict
import os
import json
import pandas as pd
from tqdm import tqdm


def read_infoseek_samples(expdir):
    qid_answer_map = defaultdict(dict)
    with open(os.path.join(expdir, "histories.json")) as f:
        histories = json.load(f)

    for history in tqdm(histories, total=len(histories), desc=f"Reading from {expdir}"):
        question_id = history[0][0]['question_id']
        question = history[0][0]['question']
        img_path = os.path.join(IMG_BASEDIR, history[0][0]['img_path'])
        answer = history[-1][0].split('[ANSWER] ')[-1]

        qid_answer_map[question_id]['question'] = question
        qid_answer_map[question_id]['answer'] = answer
        qid_answer_map[question_id]['img_path'] = img_path

    return qid_answer_map

def mark_infoseek_on_examples(qid_answer_map):
    import sys
    sys.path.append("./third_party/infoseek_eval")
    # if args.split in ['test', 'valid']:
    reference_path = f"third_party/infoseek_eval/infoseek/infoseek_val.jsonl" #NOTE M2KR "test" split = official Infoseek "val" split
    reference_qtype_path = f"third_party/infoseek_eval/infoseek/infoseek_val_qtype.jsonl"
    # elif args.split in ['train']:
    # reference_path = f"third_party/infoseek_eval/infoseek/infoseek_train.jsonl" #NOTE M2KR "test" split = official Infoseek "val" split
    # reference_qtype_path = None

    # Load M2KR Infoseek data to get all possible answers
    sys.path.append("./src")
    from vqa_datasets import load_vqa_dataset
    infoseek_ds = load_vqa_dataset(dataset_name="InfoseekNew", split="valid_m2kr")
    qid_to_all_answers = {}
    for item in infoseek_ds:
        qid_to_all_answers[item['question_id']] = item['answers']


    from infoseek_eval import evaluate_by_example, load_jsonl, prepare_qid2example
    reference = load_jsonl(reference_path)
    reference_qtype = load_jsonl(reference_qtype_path) if reference_qtype_path is not None else None
    qid2example = prepare_qid2example(reference, reference_qtype)
        # flatten data
    predictions = []
    for question_id, item in tqdm(qid_answer_map.items(), desc='evaluate example-by-example', total=len(qid_answer_map)):
        preds = [{
            'data_id': question_id,
            'prediction': item['answer']
        }]
        result, gt_answers = evaluate_by_example(preds, reference, reference_qtype, qid2example)
        score = max(result["unseen_entity_score"]["score"], result["final_score"], result["unseen_question_score"]["score"])
        qid_answer_map[question_id]['score'] = score
        qid_answer_map[question_id]['gt_answers'] = gt_answers  # Store all ground truth answers
        qid_answer_map[question_id]['all_answers'] = qid_to_all_answers[question_id]

    flat_data = []
    for question_id, item in tqdm(qid_answer_map.items(), total=len(qid_answer_map), desc='flattening data...'):
        answer, scores, img_path, gt_answers, all_answers, question = item['answer'], item['score'], item['img_path'], item['gt_answers'], item['all_answers'], item['question']
        flat_data.append((question_id, img_path, question, gt_answers, all_answers, answer, score))
        
    df = pd.DataFrame(flat_data, columns=['question_id', 'img_path', 'question', 'gt_answers', 'all_answers', 'answer', 'score'])
    return df

def gpt_judge(question, gt_answers, all_answers, answer, model="gpt-4o-mini"):
    """
    Calls the GPT-based model to evaluate the answer.
    Returns 1.0 if the answer is correct, 0.0 if incorrect.
    """
    from openai import OpenAI

    # Initialize the client with API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

    prompt = f"""
Evaluate whether the model's answer is correct given the ground truth answers and other possible acceptable answers from human annotators.
Please output ONLY 'yes' or 'no'. The answer is correct if it matches any of the ground truth or acceptable answers in meaning (not necessarily exact wording).

Question: {question}
Ground Truth Answers: {" OR ".join(gt_answers)}
Other Possible Answers: {" OR ".join(all_answers)}
Model's Answer: {answer}

Is the answer correct? Answer only 'yes' or 'no':"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a judge for question answering systems. You will only respond with 'yes' or 'no'."},
                {"role": "user", "content": prompt}
            ]
        )
        result = response.choices[0].message.content.strip().lower()
        score = 1.0 if 'yes' in result else 0.0
    except Exception as e:
        print(f"Error evaluating with GPT: {e}")
        score = 0.0  # Assign a default score in case of an error
    return score

def load_persistent_judgments(persistence_file):
    if os.path.exists(persistence_file):
        with open(persistence_file, "r") as f:
            return {entry['question_id']: entry for entry in map(json.loads, f)}
    return {}


def save_persistent_judgment(question_id, result, persistence_file):
    with open(persistence_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def evaluate_with_gpt(df, persistence_file, model="gpt-4o-mini"):
    completed_judgments = load_persistent_judgments(persistence_file)
    flat_data = []

    for row_idx, item in tqdm(df.iterrows(), desc="Evaluating with GPT", total=len(qid_answer_map)):
        question_id = item['question_id']
        if question_id in completed_judgments:
            result = completed_judgments[question_id]
        else:
            question = item['question']
            gt_answers = item['gt_answers']
            all_answers = item['all_answers']  # Now using all_answers
            answer = item['answer']
            score = gpt_judge(question, gt_answers, all_answers, answer, model)
            result = {
                'question_id': question_id,
                'question': question,
                'gt_answers': gt_answers,
                'all_answers': all_answers,  # Store all_answers in result
                'answer': answer,
                f'{model}_score': score,
                'img_path': item['img_path'],
            }
            save_persistent_judgment(question_id, result, persistence_file)

        flat_data.append((
            result['question_id'], result['img_path'], result['question'],
            result['gt_answers'], result['all_answers'], result['answer'], result[f'{model}_score']
        ))

    df = pd.DataFrame(flat_data, columns=['question_id', 'img_path', 'question', 'gt_answers', 'all_answers', 'answer', 'score'])
    return df

if __name__ == "__main__":
    for EXPDIR in EXPDIRS:
        OUTPUT_DIR=f'analysis/{OPENAI_MODEL}/{EXPDIR}'
        PERSISTENCE_FILE=f"{OUTPUT_DIR}/{OPENAI_MODEL}.jsonl"
        OUTPUT_CSV=f"{OUTPUT_DIR}/makred_results.csv"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        qid_answer_map = read_infoseek_samples(EXPDIR)
        em_df = mark_infoseek_on_examples(qid_answer_map)
        gpt_df = evaluate_with_gpt(em_df, PERSISTENCE_FILE, model=OPENAI_MODEL)
        gpt_df.to_csv(OUTPUT_CSV, index=False)
        with open(f"{OUTPUT_DIR}/score.json", 'w') as f:
            json.dump({'score': gpt_df['score'].mean() * 100}, f)  # Convert to percentage
        print(f"Results saved to {OUTPUT_CSV}")
        print("\nFinal Judgments:")
        print(gpt_df[['question_id', 'question', 'gt_answers', 'answer', 'score']].head())
        print("EXPDIR=", EXPDIR)
        print(f"{OPENAI_MODEL} judge accuracy = {gpt_df['score'].mean() * 100:.1f}%")