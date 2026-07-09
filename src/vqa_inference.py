import sys
sys.path.append('./src')

import vqa_agents
from vqa_agents import VQA_State, VQA_Environment
import _jsonnet
from easydict import EasyDict
from pprint import pprint
import json
from tqdm import tqdm
import wandb 

def make_initial_state(schema, question_id, question, img_path):
    prompt = schema.initial_prompt.replace('<<QUESTION>>', question)
    initial_state = VQA_State(
        question_id=question_id,
        question=question, 
        img_path=img_path, 
        text_context=prompt
    )
    return initial_state

def run_vqa_agent(initial_state, vqa_agent, vqa_env, seed=0, debug=False):
    """
    - initial_state: in the case of a VLM agent, this is the initial prompt (with quesions)
    - agent: must define `act` which return an action given current state
    - env: must define `state_transition` and `compute_reward`
    """
    curr_state = initial_state
    history = [(curr_state, None, 0)]
    while not curr_state.is_terminal_state:
        action = vqa_agent.act(curr_state, seed=seed)
        next_state = vqa_env.transition(curr_state, action)
        curr_state = next_state
        reward = vqa_env.compute_reward(curr_state)
        if debug:
            print(f"Step [{vqa_env.step_count}] Action:", action)
        history.append((action.response, curr_state.answer, reward, action.action_taker)) # curr_state: context; action: reward; reward=[0,1]
    return history


def run_vqa(schema, vqa_dataset, seed=0, debug=False, line_logf=None):
    """
    - schema: contain configurations of the agent (i.e., hyper-parameters for VLM/Retriever), etc, instructing the agent on how to act. Also contain how to make the initial state.
    * the aim is that this is the only thing that needs to be changed for different experiments
    - retriever
    - vlm
    - vqa_env: define the reward / state transition. 
    
    RETURN:
    - answers: answer to the quesitons 
    - history: the [(action, answer, reward, action_taker), ...] triplets that lead to the answer
    """
    answers = []
    histories = [] 

    with open(line_logf, 'a+') as lf:
        lf.seek(0)
        histories = [json.loads(row) for row in lf.readlines()]
        vqa_agent = getattr(vqa_agents, schema.agent_config.class_name)(**schema.agent_config)
        for i, item in enumerate(tqdm(vqa_dataset, total=len(vqa_dataset), desc="Inferencing")):
            if i < len(histories):
                continue
            question_id, question, img, gt_answer = item['question_id'], item['question'], item['img_path'], item['gold_answer']
            vqa_env = VQA_Environment(gt_answer=gt_answer, question_id=question_id)

            initial_state = make_initial_state(schema, question_id, question, img)
            history = run_vqa_agent(initial_state, vqa_agent, vqa_env, seed=seed, debug=debug)
            ans = history[-1][1] # extract answer from terminal state

            answers.append(ans)
            histories.append(history)
            lf.write(json.dumps(history, default=lambda o: o.__dict__)+'\n')

    return answers, histories

import argparse
from datetime import datetime
import os
from vqa_datasets import load_vqa_dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--continue_expdir", type=str, default=None)
    parser.add_argument("--img_basedir", type=str, default='data')
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--config_file", type=str, default='config/config.jsonnet')
    parser.add_argument("--output_dir", type=str, default='outputs')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_cases", nargs="+")

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="a-ravqa", name=args.exp_name, entity="byrne-lab")
 
    current_time = datetime.now().strftime("%Y%m%d-%H")
    output_dir = args.continue_expdir or f"{args.output_dir}/{current_time}-{args.exp_name}"
    print("output_dir=",output_dir)
    os.makedirs(output_dir, exist_ok=True)

    line_logf = f"{output_dir}/histories.jsonl"
    test_schema = json.loads(_jsonnet.evaluate_file(args.config_file))
    with open(test_schema['initial_prompt'], 'r') as f:
        test_schema['initial_prompt'] = f.read()
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(test_schema, f)
    test_schema = EasyDict(test_schema)
    
    vqa_dataset = load_vqa_dataset(args.dataset_name, split=args.split, img_basedir=args.img_basedir)
    if args.debug:
        if args.debug_cases:
            vqa_dataset = vqa_dataset.select([int(i) for i in args.debug_cases])
        else:
            vqa_dataset = vqa_dataset.select([i for i in range(10)])

    answers, histories = run_vqa(test_schema, vqa_dataset, debug=args.debug, line_logf=line_logf)
    with open(f"{output_dir}/histories.json", 'w') as f:
        json.dump(histories, f, default=lambda o: o.__dict__)
    with open(f"{output_dir}/answers.json", 'w') as f:
        json.dump(answers, f, default=lambda o: o.__dict__)
    print(f"Done! Results saved to {output_dir}")

    wandb.finish()

#         'agent_config': {
#             'class_name': 'RAG_VQA_Agent',
#             'vlm_class': 'QWen2VLM',
#             'vlm_config': {
#                 'model_path': 'QWen/QWen2-VL-7B-Instruct',
#                 'generation_config': {'temperature': 0.3, 'max_new_tokens': 512}
#             },
#             'retriever_class': 'DummyRetriever',
#             'retriever_config': {
#                 'always_return': "Giant hogweed is a member of the carrot family and its resemblance to Queen Anne’s lace caused it to become a garden ornamental. It spreads easily and can establish along roadsides, ditches, and streams. Giant hogweed has a thick bright green stem (3-8 cm in diameter) with dark reddish-purple spots and coarse white hairs at the base of the leaf stock. The plant can be 2-5.5 m tall with broad leaves that are deeply-lobed and serrated. From late spring to mid-summer, giant hogweed produces a large upside-down umbrella-shaped head, up to 80 cm across, with clusters of tiny white flowers. Giant hogweed has a phototoxic sap that, when exposed to light, can cause severe burns on human skin. Removing hogweed can be dangerous because of this sap; it should also not be burned or composted for this reason. The easiest way to remove giant hogweed is to pull it when it is still very young and small and store all plant components in sealed black garbage bags until the plant is dried and seeds are no longer viable. Do not plant giant hogweed in gardens and report any sightings."
#             }
#         },
#         'initial_prompt': 
#             """Answer the question about the image. You should respond in the following schema. You must generate in bullet points with title [THINK], [RETRIEVE], or [ANSWER].\n\
# - For [THINK] bullet, you should reason about the question given the context\n\
# - For [RETRIEVE] bullet, you should generate the query text for a retriever to retrieve relevant document from the database. You should stop generation on completion of the [RETRIEVE] bullet.\n\
# - For [ANSWER] bullet, you should provide the final answer to the question. Your answer should be brief and assertive.\n\
# The retriever will return the retrived document in a bullet titled [EVIDENCE].You should continue generation after you received the evidence.\n\
# Question: <<QUESTION>>\n\
# Response:
#             """,
#     }