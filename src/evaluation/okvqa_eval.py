import sys
sys.path.append('./src')
sys.path.append('./src/evaluation')
from vqa_datasets import load_vqa_dataset
import numpy as np
from easydict import EasyDict
from tqdm import tqdm

import wandb
import logging
logger = logging.getLogger(__name__)

from vqaEval import VQAEval
from vqa_tools import VQA
import argparse
import json 

#from utils.dirs import *
question_path = "/mnt/g/Datasets/OKVQA/OpenEnded_mscoco_val2014_questions.json"
annotation_path = "/mnt/g/Datasets/OKVQA/mscoco_val2014_annotations.json"


def compute_okvqa_scores(answers) -> dict:
    """
    Compute OkVQA scores
    """
    metrics_to_log = {}
    ##############################
    ##    Compute VQA Scores    ##
    ##############################
    #mode = data_dict['mode']
    #answers = data_dict['batch_predictions']

    #torch.distributed.barrier()
    #num_processes = torch.distributed.get_world_size()

    #if not os.path.exists(self.config.ckpt_dir):
    #    create_dirs([self.config.ckpt_dir])

    # save tmp files for each process
    #tmp_dir = os.path.join(self.config.ckpt_dir, f"tmp_{self.global_rank}.pkl")
    #with open(tmp_dir, 'wb') as f:
    #    pickle.dump(answers, f)
    #logger.info(f"Save tmp file {tmp_dir} for process {self.global_rank}.")
    
    #torch.distributed.barrier()
    # load tmp files for each process
    #all_answers = []
    #for i in range(num_processes):
    #    tmp_dir = os.path.join(self.config.ckpt_dir, f"tmp_{i}.pkl")
    #    with open(tmp_dir, 'rb') as f:
    #        all_answers.extend(pickle.load(f))
    #    logger.info(f"Load tmp file {tmp_dir} for process {i}.")
    
    #torch.distributed.barrier()
    #logger.info(f"extended answers from {len(answers)} to {len(all_answers)}")
    #answers = all_answers

    # create vqa object and vqaRes object
    # These are the question files and annotation files of okvqa 
    
    # Note that the answer need to be in the format of list of dicts: 
    # predictions.append({
    #            'question_id': question_id,
    #            'answer': decoded_output,
    #        }) 
    
    
    
    predictions = []
    vqa_dataset  = load_vqa_dataset("OKVQA", split="valid", img_basedir='data/')
    
    question_ids = vqa_dataset["question_id"]
    print(question_ids[0:10], len(question_ids))
    print(len(answers))
    
    for i, ans in enumerate(answers):
        predictions.append({
            'question_id': int(question_ids[i]),
            'answer': ans
        })
    
    vqa_helper = VQA(annotation_path, question_path)
    vqaRes = vqa_helper.loadResFromDict(predictions)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa_helper, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate()

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


if __name__ == '__main__':
    #answers = evaluate.load_predictions(args.prediction_file)
    #log_dict = evaluate.compute_okvqa_scores(answers, log_dict)
    #evaluate.save_log_dict(log_dict, args.output_dir)
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction_file", type=str)
    args = parser.parse_args()
    # Read the txt files and convert to list of strings 
    with open(args.prediction_file, 'r') as f:
        answers = json.load(f) 
        
    print(len(answers))
    
    compute_okvqa_scores(answers)