import sys
sys.path.append('./src')
from dataclasses import dataclass
from copy import deepcopy
from abc import ABC, abstractmethod
import vlms
import retrievers



@dataclass
class VQA_State:
    question: str
    text_context: str
    img_path: str
    is_terminal_state: bool = False
    answer: str = None
    question_id: int = 0

    def __repr__(self):
        return f"id={self.question_id} |||| question={self.question} |||| img_path={self.img_path} |||| answer={self.answer} |||| text_context={self.text_context} |||| is_terminal_state={self.is_terminal_state}"

    def dict(self):
        return {
            "question": self.question,
            "text_context": self.text_context,
            "img_path": self.img_path,
            "is_terminal_state": self.is_terminal_state,
        }


@dataclass
class VQA_Action:
    response: str
    action_taker: str = 'agent'
    action_id: str = '0'

    def __repr__(self):
        return f"{self.action_id}, {self.action_taker}, {self.response}"
    

@dataclass
class VQA_Environment:
    gt_answer: str
    question_id: str = "0"
    step_count: int = 0

    def transition(self, state, action):
        new_state = deepcopy(state)
        new_state.text_context += f"\n{action.response}"
        if '[ANSWER]' in action.response:
            new_state.is_terminal_state = True
            new_state.answer = self.extract_answer(action.response)
        # elif not any([x in action.response for x in ['[THINK]', '[EVIDENCE]', '[RETRIEVE]']]):
        #     new_state.is_terminal_state = True
        #     new_state.answer = "Fail to generate answer"
        elif self.step_count >= 6: 
            new_state.is_terminal_state = True
            new_state.answer = "Fail to generate answer (Maxmimal step exceeded)"
        else:
            pass
        self.step_count += 1
        return new_state
    
    def extract_answer(self, text):
        ans = "[ANSWER]".join(text.split('[ANSWER]')[1:])
        return ans
    
    def eval_answer_is_correct(self, ans):
        """
        For the time-being, use exact match
        """
        return ans == self.gt_answer
        
    def compute_reward(self, state):
        return self.eval_answer_is_correct(state.answer)

class VQA_Agent(ABC):
    def __init__(self, class_name, *args, **kwargs):
        self.class_name = class_name
    
    @abstractmethod
    def act(self, state, seed):
        pass

class Plain_VQA_Agent(VQA_Agent):
    def __init__(self, vlm_class, vlm_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vlm = getattr(vlms, vlm_class)(**vlm_config)
    
    def act(self, state, seed=0):
        input_text, input_img = state.text_context, state.img_path
        response = self.vlm.generate_response(input_text=input_text, input_img=input_img, seed=0)
        response = "[ANSWER]" + response
        action = VQA_Action(response)
        return action

class ConventionalRAG_VQA_Agent(VQA_Agent):
    def __init__(self, vlm_class, vlm_config, retriever_class, retriever_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vlm = getattr(vlms, vlm_class)(**vlm_config)
        self.retriever = getattr(retrievers, retriever_class)(**retriever_config)
    
    def extract_query_from_state(self, state):
        query = None
        bullets = state.text_context.split('\n')
        last_bullet = bullets[-1]
        second_last_bullet = bullets[-2]
        if last_bullet.startswith('Question: ') or second_last_bullet.startswith('Question: '):
            query = last_bullet.split('Question: ')[1]
        return query
    
    def retriever_action(self, retrieved_doc):
        if isinstance(retrieved_doc, str):
            retriever_action = f"\n[EVIDENCE] {retrieved_doc}\n"
        elif isinstance(retrieved_doc, list):
            retriever_action = ""
            for i, doc in enumerate(retrieved_doc):
                retriever_action += f"\n[EVIDENCE {i}] {doc}\n" 
        else:
            raise NotImplementedError("retriever_action")
        return retriever_action

    def act(self, state, seed=0):
        input_text, input_img = state.text_context, state.img_path
        query = self.extract_query_from_state(state)
        if query is not None:
            retrieved_doc = self.retriever.query(query, input_img, question_id=state.question_id)
            response = self.retriever_action(retrieved_doc)
            action = VQA_Action(response, action_taker='retriever')
        else:
            response = self.vlm.generate_response(input_text=input_text, input_img=input_img)
            response = "[ANSWER]" + response
            action = VQA_Action(response, action_taker='vlm')
        return action

class RAG_VQA_Agent(VQA_Agent):
    def __init__(self, vlm_class, vlm_config, retriever_class, retriever_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vlm = getattr(vlms, vlm_class)(**vlm_config)
        self.retriever = getattr(retrievers, retriever_class)(**retriever_config)
    
    def extract_query_from_state(self, state):
        query = None
        bullets = state.text_context.split('\n')
        last_bullet = bullets[-1]
        if last_bullet.startswith('[RETRIEVE]'):
            query = last_bullet.split('[RETRIEVE]')[1]
        return query
    
    # def merge_retrieved_doc(self, retrieved_doc, input_text, input_img):
    #     retriever_action = f"\n[EVIDENCE] {retrieved_doc}\n"
    #     return retriever_action
    
    def retriever_action(self, retrieved_doc):
        retriever_action = f"\n[EVIDENCE] {retrieved_doc}\n"
        return retriever_action

    def act(self, state, seed=0):
        input_text, input_img = state.text_context, state.img_path
        query = self.extract_query_from_state(state)
        if query is not None:
            retrieved_doc = self.retriever.query(query, input_img, question_id=state.question_id)
            response = self.retriever_action(retrieved_doc)
            action_taker = 'retriever'
        else:
            response = self.vlm.generate_response(input_text=input_text, input_img=input_img)
            action_taker = 'vlm'
        action = VQA_Action(response, action_taker=action_taker)
        return action

class SearchEngineInterface_VQA_Agent(VQA_Agent):
    def __init__(self, vlm_class, vlm_config, retriever_class, retriever_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vlm = getattr(vlms, vlm_class)(**vlm_config)
        self.retriever = getattr(retrievers, retriever_class)(**retriever_config)
    
    def extract_query_from_state(self, state):
        query = None
        bullets = state.text_context.split('\n')
        last_bullet = bullets[-1]
        if last_bullet.lstrip().startswith('RETRIEVE_SEARCH'):
            query = ('search', last_bullet.split('RETRIEVE_SEARCH(')[1].rstrip(')'))
        elif last_bullet.lstrip().startswith('RETRIEVE_VIEW'):
            query = ('view', int(last_bullet.split('RETRIEVE_VIEW(' )[1].rstrip(')')))
        elif last_bullet.lstrip().startswith('RETRIEVE_NEXTPAGE'):
            query = ('next', None)
        return query
    
    # def merge_retrieved_doc(self, retrieved_doc, input_text, input_img):
    #     retriever_action = f"\n[EVIDENCE] {retrieved_doc}\n"
    #     return retriever_action
    
    def retriever_action(self, query, retrieved_docs):
        if query[0] == 'search' or query[0] == 'next':
            ret_text = [f"[PREVIEW DOCUMENT {doc['idx']}] Title={doc['title']}. {doc['text']}" for doc in retrieved_docs]
            ret_text = '\t\n'.join(ret_text)
        elif query[0] == 'view':
            doc = retrieved_doc[0]
            ret_text = f"\t\n[FULL DOCUMENT {doc['idx']}] {doc['text']}\n"
        else:
            raise NotImplementedError(f"retriever action")
        return retriever_action

    def act(self, state, seed=0):
        input_text, input_img = state.text_context, state.img_path
        query = self.extract_query_from_state(state)
        if query is not None:
            action_taker = 'retriever'
            if query[0] == 'search':
                retrieved_docs = self.retriever.query(query[1], input_img, question_id=state.question_id)
                response = self.retriever_action(query, retrieved_doc)
            elif query[0] == 'view':
                retrieved_docs = self.retriever.view_fulltext(query[1])
                response = self.retriever_action(query, retrieved_docs)
            elif query[0] == 'next':
                retrieved_docs = self.retriever.next()
                response = self.retriever_action(query, retrived_docs)
        else:
            response = self.vlm.generate_response(input_text=input_text, input_img=input_img)
            action_taker = 'vlm'
        action = VQA_Action(response, action_taker=action_taker)
        return action

if __name__ == '__main__':
    action = VQA_Action("Some text from LLM")
    state = VQA_State("?", "prompt", "/extra/")
    print((action, state))
    
