import sys
sys.path.append('./src')
sys.path.append('./src/ops')
import _jsonnet
import vqa_agents
from vqa_agents import VQA_State, VQA_Environment
from pipeline import SequentialPipeline

from easydict import EasyDict
from pprint import pprint
import json
from tqdm import tqdm
import wandb 
from pprint import pprint

def make_initial_state(schema, question_id, question, img_path):
    prompt = schema.initial_prompt.replace('<<QUESTION>>', question)
    initial_state = VQA_State(
        question_id=question_id,
        question=question, 
        img_path=img_path, 
        text_context=prompt
    )
    return initial_state

import argparse
from datetime import datetime
import os
from vqa_datasets import load_vqa_dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--take_n", type=int, default=-1)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--continue_expdir", type=str, default=None)
    parser.add_argument("--img_basedir", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ds_seed", type=int, default=0)

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None)

    parser.add_argument("--config_file", type=str, default='config/config.jsonnet')
    parser.add_argument("--output_dir", type=str, default='outputs')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_cases", nargs="+")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--retrieve_topk", type=int, default=None)
    parser.add_argument("--rerank_topk", type=int, default=None)

    parser.add_argument("--retrieval_ds_path", type=str, default=None)
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="a-ravqa", name=args.exp_name, entity="byrne-lab")
 
    current_time = datetime.now().strftime("%Y%m%d-%H")
    if args.continue_expdir is None or len(args.continue_expdir) == 0:
        args.continue_expdir = None
    output_dir = args.continue_expdir or f"{args.output_dir}/{current_time}-{args.exp_name}"
    if args.debug:
        output_dir += '-debug'
    print("output_dir=",output_dir)
    os.makedirs(output_dir, exist_ok=True)

    line_logf = f"{output_dir}/histories.jsonl"
    test_schema = json.loads(_jsonnet.evaluate_file(args.config_file))
    if args.model_path is not None:
        test_schema['agent_config']['vlm_config']['model_path'] = args.model_path
    if args.base_model_path is not None and args.base_model_path != "":
        test_schema['agent_config']['vlm_config']['base_model_path'] = args.base_model_path
    else:
        test_schema['agent_config']['vlm_config']['is_lora'] = False
        print(f"is_lora set to False because no args.base_model_path is provided!")
    if args.retrieve_topk is not None:
        for op_config in test_schema['op_config']:
            if 'Retrieve' in op_config['name']:
                op_config['kwargs']['ret_topk'] = args.retrieve_topk
                print(f"overwrite ret_topk={args.retrieve_topk} for {op_config['name']}")
    if args.rerank_topk is not None:
        for op_config in test_schema['op_config']:
            if 'RetrieveRerank' in op_config['name']:
                op_config['kwargs']['rerank_topk'] = args.rerank_topk
                print(f"overwrite ret_topk={args.rerank_topk} for {op_config['name']}")
    if args.retrieval_ds_path is not None:
        test_schema['agent_config']['retriever_config']['ds_path'] = args.retrieval_ds_path        
    with open(test_schema['initial_prompt'], 'r') as f:
        test_schema['initial_prompt'] = f.read()
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(test_schema, f)
    test_schema = EasyDict(test_schema)
    pprint(test_schema)

    vqa_dataset = load_vqa_dataset(args.dataset_name, split=args.split, img_basedir=args.img_basedir, take_n=args.take_n, seed=args.ds_seed)
    if args.debug:
        if args.debug_cases:
            vqa_dataset = vqa_dataset.select([int(i) for i in args.debug_cases])
        else:
            vqa_dataset = vqa_dataset.select([i for i in range(10)])

    vqa_pipeline = SequentialPipeline(
        agent_config=test_schema['agent_config'],
        op_config=test_schema['op_config'],
        prompt_template=test_schema['initial_prompt'],
        output_dir=output_dir
    )

    print("args.seed = ",args.seed)
    answers, histories = vqa_pipeline.run_vqa(vqa_dataset, debug=args.debug, line_logf=line_logf, seed=args.seed, override=args.override)

    vqa_pipeline.finalize()

    with open(f"{output_dir}/histories.json", 'w') as f:
        json.dump(histories, f, default=lambda o: o.__dict__)
    with open(f"{output_dir}/answers.json", 'w') as f:
        json.dump(answers, f, default=lambda o: o.__dict__)
    print(f"Done! Results saved to {output_dir}")

    if args.do_eval:
        if args.dataset_name == 'EVQA':
            sys.path.append('./src/evaluation')
            df = vqa_dataset.to_pandas()

            from evqa_eval_1004 import process_row as eval_process_row
            from evqa_eval_1004 import process_row_mp as eval_process_row_mp
            from evqa_eval_1004 import extract_queries_and_retrieved_docs

            if len(answers) != len(histories):
                answers = [hist[-1][0].split('[ANSWER] ')[1] for hist in histories]
            # queries_and_docs_and_calls = [extract_queries_and_retrieved_docs(hist) for hist in histories]
            # df['queries'] = [qd[0] for qd in queries_and_docs_and_calls]
            # df['retrieved_docs'] = [qd[1] for qd in queries_and_docs_and_calls] 
            df['prediction'] = answers
            all_eval_results = []
            import tensorflow as tf
            if tf.test.is_gpu_available():
                for row in tqdm(df.itertuples(), total=len(df)):
                    eval_result = eval_process_row(row)
                    all_eval_results.append(eval_result)
                dict_to_report = {f"avg_{k}": sum([res[k] for res in all_eval_results])/len(all_eval_results) for k in all_eval_results[0]}
                for k in all_eval_results[0]:
                    df[k] = [res[k] for res in all_eval_results]
            else:
                import multiprocessing
                with multiprocessing.Pool(processes=8) as pool:
                    all_eval_results = list(tqdm(pool.imap(eval_process_row_mp, df.iterrows(), chunksize=1), total=len(df)))
                dict_to_report = {f"avg_score": sum(all_eval_results)/len(all_eval_results)}
                df['score'] = all_eval_results
            
            df.to_csv(f"{output_dir}/marked_answers.csv")
            with open(f'{output_dir}/scores.json', 'w') as f:
                json.dump(dict_to_report, f)
            print("Evaluation results saved to", output_dir)
        elif args.dataset_name == 'OKVQA':
            sys.path.append('./src/evaluation')
            from vqaEval import VQAEval
            from vqa_tools import VQA
            df = vqa_dataset.to_pandas()

            answers = [hist[-1][0].split('[ANSWER] ')[1] for hist in histories]

            #NOTE JC
            if args.split == 'valid' or args.split == 'test':
                question_path = "../vqa_data/KBVQA_data/ok-vqa/OpenEnded_mscoco_val2014_questions.json"
                annotation_path = "../vqa_data/KBVQA_data/ok-vqa/mscoco_val2014_annotations.json"
            elif args.split == 'train':
                question_path = "../vqa_data/KBVQA_data/ok-vqa/OpenEnded_mscoco_train2014_questions.json"
                annotation_path = "../vqa_data/KBVQA_data/ok-vqa/mscoco_train2014_annotations.json"
            else:
                raise NotImplementedError("OKVQA annotations")

            vqa_helper = VQA(annotation_path, question_path)
            predictions = []
            all_question_ids = []
            for i, item in enumerate(df.itertuples()):
                predictions.append({
                    'question_id': int(item.question_id),
                    'answer': answers[i]
                })
                all_question_ids.append(int(item.question_id))

            vqaRes = vqa_helper.loadResFromDict(predictions)
            vqaEval = VQAEval(vqa_helper, vqaRes, n=2)
            vqaEval.evaluate()

            metrics_to_log = {}
            # print accuracies
            print ("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
            print ("Per Question Type Accuracy is the following:")
            for quesType in vqaEval.accuracy['perQuestionType']:
                print ("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
            print ("\n")
            print ("Per Answer Type Accuracy is the following:")
            for ansType in vqaEval.accuracy['perAnswerType']:
                print ("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
            print ("\n")

            metrics_to_log['accuracy_overall'] = vqaEval.accuracy['overall']
            for quesType in vqaEval.accuracy['perQuestionType']:
                metrics_to_log[f'accuracy_QuestionType_{quesType}'] = vqaEval.accuracy['perQuestionType'][quesType]
            for ansType in vqaEval.accuracy['perAnswerType']:
                metrics_to_log[f'accuracy_AnswerType_{ansType}'] = vqaEval.accuracy['perAnswerType'][ansType]
            
            with open(f'{output_dir}/scores.json', 'w') as f:
                json.dump(metrics_to_log, f)
            print("Evaluation results saved to", output_dir)

        elif args.dataset_name == 'Infoseek' or args.dataset_name == "InfoseekNew":
            sys.path.append("./third_party/infoseek_eval")
            predictions = []
            df = vqa_dataset.to_pandas()

            answers = [hist[-1][0].split('[ANSWER] ')[1] for hist in histories]

            for i, item in enumerate(df.itertuples()):
                predictions.append({
                    'data_id': item.question_id,
                    'prediction': answers[i]
                })
            
            pred_path = f"{output_dir}/predictions.jsonl"
            with open(pred_path, 'w') as f:
                for pred in predictions:
                    f.write(json.dumps(pred)+'\n')
            if args.split in ['test', 'valid', 'valid_m2kr']:
                reference_path = f"third_party/infoseek_eval/infoseek/infoseek_val.jsonl" #NOTE M2KR "test" split = official Infoseek "val" split
                reference_qtype_path = f"third_party/infoseek_eval/infoseek/infoseek_val_qtype.jsonl"
            elif args.split in ['train']:
                reference_path = f"third_party/infoseek_eval/infoseek/infoseek_train.jsonl" #NOTE M2KR "test" split = official Infoseek "val" split
                reference_qtype_path = None

            from infoseek_eval import evaluate
            result = evaluate(pred_path, reference_path, reference_qtype_path)
            final_score = result["final_score"]
            unseen_question_score = result["unseen_question_score"]["score"]
            unseen_entity_score = result["unseen_entity_score"]["score"]
            print(f"{args.split} final score: {final_score}")
            print(f"{args.split} unseen question score: {unseen_question_score}")
            print(f"{args.split} unseen entity score: {unseen_entity_score}")
            
            dict_to_report = {
                'score': final_score,
                'unseen_question_score': unseen_question_score,
                'unseen_entity_score': unseen_entity_score
            }
            with open(f'{output_dir}/scores.json', 'w') as f:
                json.dump(dict_to_report, f)
            print("Evaluation results saved to", output_dir)
            
        elif args.dataset_name == 'OKVQA-heldout':
            sys.path.append('./src/evaluation')
            df = vqa_dataset.to_pandas()
            df['question_type'] = 'automatic'

            from evqa_eval_1004 import process_row as eval_process_row
            from evqa_eval_1004 import process_row_mp as eval_process_row_mp
            from evqa_eval_1004 import extract_queries_and_retrieved_docs

            if len(answers) != len(histories):
                answers = [hist[-1][0].split('[ANSWER] ')[1] for hist in histories]
            # queries_and_docs_and_calls = [extract_queries_and_retrieved_docs(hist) for hist in histories]
            # df['queries'] = [qd[0] for qd in queries_and_docs_and_calls]
            # df['retrieved_docs'] = [qd[1] for qd in queries_and_docs_and_calls] 
            df['prediction'] = answers
            all_eval_results = []
            import tensorflow as tf
            if tf.test.is_gpu_available():
                for row in tqdm(df.itertuples(), total=len(df)):
                    eval_result = eval_process_row(row)
                    all_eval_results.append(eval_result)
                dict_to_report = {f"avg_{k}": sum([res[k] for res in all_eval_results])/len(all_eval_results) for k in all_eval_results[0]}
                for k in all_eval_results[0]:
                    df[k] = [res[k] for res in all_eval_results]
            else:
                import multiprocessing
                with multiprocessing.Pool(processes=8) as pool:
                    all_eval_results = list(tqdm(pool.imap(eval_process_row_mp, df.iterrows(), chunksize=1), total=len(df)))
                dict_to_report = {f"avg_score": sum(all_eval_results)/len(all_eval_results)}
                df['score'] = all_eval_results
            
            df.to_csv(f"{output_dir}/marked_answers.csv")
            with open(f'{output_dir}/scores.json', 'w') as f:
                json.dump(dict_to_report, f)
            print("Evaluation results saved to", output_dir)
        else:
            raise NotImplementedError(f"Evaluation not implemented for {args.dataset_name}")

    wandb.finish()