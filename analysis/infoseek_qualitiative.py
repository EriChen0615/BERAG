# EXPDIR="outputs/20250121-15-InfoseekNew_valid_m2kr-256_OracleRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_rag1-ft_ckpt-2000_PreFLMR-L"
# EXPDIR="outputs/20250122-15-InfoseekNew_valid_m2kr-256_OracleRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_pretrained_PreFLMR-L"
EXPDIR="outputs/20250122-15-InfoseekNew_valid_m2kr-256_OracleRetrieve[TopK]-Read_RetrieveTopK=1_QWen2VL-7B-LoRA_pretrained_PreFLMR-L"
IMG_BASEDIR = ''
OPENAI_MODEL='gpt-4o-mini-2024-07-18'
OUTPUT_DIR=f'analysis/{OPENAI_MODEL}/{EXPDIR}'
PERSISTENCE_FILE=f"{OUTPUT_DIR}/{OPENAI_MODEL}.jsonl"
OUTPUT_CSV=f"{OUTPUT_DIR}/marked_results.csv"

from collections import defaultdict
import os
import json
import pandas as pd
from tqdm import tqdm


def read_infoseek_samples(expdir):
    qid_answer_map = defaultdict(dict)
    with open(os.path.join(expdir, "histories.json")) as f:
        histories = json.load(f)
    with open(os.path.join(expdir, "VLMReadEvidence_histories.json")) as f:
        vlm_read_histories = json.load(f)

    for history, vlm_history in tqdm(zip(histories, vlm_read_histories), total=len(histories), desc=f"Reading from {expdir}"):
        question_id = history[0][0]['question_id']
        question = history[0][0]['question']
        img_path = os.path.join(IMG_BASEDIR, history[0][0]['img_path'])
        answer = history[-1][0].split('[ANSWER] ')[-1]

        qid_answer_map[question_id]['question'] = question
        qid_answer_map[question_id]['answer'] = answer
        qid_answer_map[question_id]['img_path'] = img_path
        qid_answer_map[question_id]['prompt'] = vlm_history[3]['input_text']

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
        qid_answer_map[question_id]['gt_answer'] = gt_answers[0]
                
    flat_data = []
    for question_id, item in tqdm(qid_answer_map.items(), total=len(qid_answer_map), desc='flattening data...'):
        answer, scores, prompt, img_path, gt_answer, question = item['answer'], item['score'], item['prompt'], item['img_path'], item['gt_answer'], item['question']
        flat_data.append((question_id, img_path, question, gt_answer, answer, score, prompt))
        
    df = pd.DataFrame(flat_data, columns=['question_id', 'img_path', 'question', 'gt_answer', 'answer', 'score', 'prompt'])
    return df

def gpt_judge(question, gt_answer, answer, model="gpt-4o-mini"):
    """
    Calls the GPT-based model to evaluate the answer.
    Returns a score between 0.0 and 100.0.
    """
    import openai

    # Ensure the OpenAI API key is set in the environment
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

    prompt = f"""
Evaluate the model's answer to a question, considering the ground truth answer. 
Provide a score between 0.0 and 100.0, where 100.0 means the model's answer is correct without ambiguity.
Please output your score after 'Score:'. You do not need to provide any explanation. 

Question: {question}
Ground Truth Answer: {gt_answer}
Model's Answer: {answer}

"""

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "You are a judge for question answering systems."},
                      {"role": "user", "content": prompt}]
        )
        print(response)
        score = float(response['choices'][0]['message']['content'].split('Score:')[-1].strip())
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
            gt_answer = item['gt_answer']
            answer = item['answer']
            score = gpt_judge(question, gt_answer, answer, model)
            result = {
                'question_id': question_id,
                'question': question,
                'gt_answer': gt_answer,
                'answer': answer,
                f'{model}_score': score,
                'img_path': item['img_path'],
                'prompt': item['prompt']
            }
            save_persistent_judgment(question_id, result, persistence_file)

        flat_data.append((
            result['question_id'], result['img_path'], result['question'],
            result['gt_answer'], result['answer'], result[f'{model}_score'], result['prompt']
        ))

    df = pd.DataFrame(flat_data, columns=['question_id', 'img_path', 'question', 'gt_answer', 'answer', 'score', 'prompt'])
    return df

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    qid_answer_map = read_infoseek_samples(EXPDIR)
    em_df = mark_infoseek_on_examples(qid_answer_map)
    gpt_df = evaluate_with_gpt(em_df, PERSISTENCE_FILE, model=OPENAI_MODEL)
    gpt_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")
    print("\nFinal Judgments:")
    print(gpt_df[['question_id', 'question', 'gt_answer', 'answer', 'score']].head())
    print(f"{OPENAI_MODEL} judge score = ", gpt_df['score'].mean())

    breakpoint()
    pass