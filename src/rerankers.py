from abc import ABC, abstractmethod
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from PIL import Image
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import os
import json


class All_Reranker(ABC):
    @abstractmethod
    def rank(self, question, query_img, retrieved_docs):
        pass
    
class VLMReranker(All_Reranker):
    def __init__(
        self,
        model_path,
        prompt_template_file,
        *args,
        is_lora=False,
        is_cls=False,
        base_model_path=None,
        processor_path=None,
        reranker_bz=4,
        load_4bit=False,
        load_8bit=False,
        **kwargs
    ):
        self.model_path = model_path
        self.is_lora = is_lora
        self.is_cls = is_cls
        self.base_model_path = base_model_path
        self.processor_path = processor_path
        with open(prompt_template_file, 'r') as f:
            self.prompt_template = f.read()

        self.model = None
        self.bz = reranker_bz
        self.load_4bit = load_4bit
        self.load_8bit = load_8bit

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
class QWen2Reranker(VLMReranker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _init_model(self):
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        if self.load_4bit:
            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype="bfloat16",
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_quant_storage="bfloat16",  # crucial for fsdp+qlora
                                )
            print("4-bit quantization enabled")
        elif self.load_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("8-bit quantization enabled")
        else:
            quantization_config = None
        
        if not self.is_lora:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                quantization_config=quantization_config,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(self.processor_path or self.model_path)
        else:
            assert self.base_model_path is not None
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                quantization_config=quantization_config,
            ).eval()
            self.model.load_adapter(self.model_path)
            self.processor = AutoProcessor.from_pretrained(self.base_model_path)

    @torch.no_grad()
    def get_next_token_ps(self, query_text, query_img):
        if self.model is None:
            self._init_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": query_img},
                    {"type": "text", "text": query_text},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        y_nexts = ["Yes", "No"]

        y_nexts_idx = self.processor(
            text=y_nexts,
        )['input_ids']

        y_next_idx_map = {yn: y_idx[0] for yn, y_idx in zip(y_nexts, y_nexts_idx)}

        # Inference
        self.model.eval()
        outputs = self.model(**inputs)
        last_logits = outputs.logits[0][-1]
        m = torch.nn.Softmax(dim=0)

        last_probs = m(last_logits)
        output_probs = {
            yn: last_probs[idx].item() for yn, idx in y_next_idx_map.items()
        }
        return output_probs
    
    def rank(self, question, query_img, retrieved_docs):
        all_scores = []
        for this_doc in retrieved_docs:
            query_text = self.prompt_template\
                            .replace('<<EVIDENCE>>', this_doc['text'])\
                            .replace('<<QUESTION>>', question)
            next_token_ps = self.get_next_token_ps(query_text, query_img)
            score = next_token_ps['Yes']
            this_doc['rerank_score'] = score
            all_scores.append(score)
        
        reranked_documents = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
        return reranked_documents
        
        
        
class QWen2CLSReranker(QWen2Reranker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _init_cls(self):
        classifier_config_path = os.path.join(self.model_path, 'classifier_config.json')
        classifier_bin_path = os.path.join(self.model_path, 'classifier.bin')
        assert os.path.exists(classifier_config_path)
        assert os.path.exists(classifier_bin_path)
        
        with open(classifier_config_path, "r") as f:
            classifier_config = json.load(f)
        classifier = Classifier(
            input_shape=classifier_config["hidden_size"],
            num_layers=classifier_config["num_layers"],
            proj_dim=classifier_config["proj_dim"],
            output_dim=classifier_config["output_dim"],
            input_dropout=classifier_config["input_dropout"],
            dropout=classifier_config["dropout"]
        )
        self.model.add_module("classifier", classifier.to("cuda").to(torch.bfloat16).eval())
        self.model.classifier.load_state_dict(torch.load(classifier_bin_path,  weights_only=True))
        print("Classifier loaded")
        #print(self.model.classifier)

    @torch.no_grad()
    def get_next_token_ps_bz1(self, query_text, query_img):
        if self.model is None:
            self._init_model()
            if self.is_cls:
                self._init_cls()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": query_img},
                    {"type": "text", "text": query_text},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # Inference
        self.model.eval()
        outputs = self.model(**inputs, output_hidden_states=True)

        embed = outputs.hidden_states[-1][:, -1, :]
        output_logits = self.model.classifier(embed)
        output_probs = torch.sigmoid(output_logits)

        return output_probs[0]
    
    def rank_bz1(self, question, query_img, retrieved_docs):
        all_scores = []
        for this_doc in retrieved_docs:
            query_text = self.prompt_template\
                            .replace('<<EVIDENCE>>', this_doc['text'])\
                            .replace('<<QUESTION>>', question)
            next_token_ps = self.get_next_token_ps(query_text, query_img)
            score = next_token_ps
            this_doc['rerank_score'] = score
            all_scores.append(score)
        
        reranked_documents = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
        return reranked_documents
    
    # Write batching version
    @torch.no_grad()
    def get_next_token_ps(self, query_texts, query_imgs):
        if self.model is None:
            self._init_model()
            # Assert the processor is left padded
            assert self.processor.tokenizer.padding_side == "left" 
            self.yes_token_id = self.processor.tokenizer.convert_tokens_to_ids("Yes")
            self.no_token_id = self.processor.tokenizer.convert_tokens_to_ids("No")
            if self.is_cls:
                self._init_cls()

        messages_batch = []
            
        for query_text, query_img in zip(query_texts, query_imgs):

            messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": query_img},
                    {"type": "text", "text": query_text},
                ],
            }
            ]
            messages_batch.append(messages)
        
        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) for messages in messages_batch
        ]
        #print(texts)
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        self.model.eval()
        outputs = self.model(**inputs, output_hidden_states=True)
        
        embed = outputs.hidden_states[-1][:, -1, :]
        output_logits = self.model.classifier(embed)
        output_probs = torch.sigmoid(output_logits)
        
        return output_probs
    
    def rank(self, question, query_img, retrieved_docs):
        all_scores = []
        batch_texts = []
        batch_imgs = []
        batch_docs = []
        for this_doc in retrieved_docs:
            query_text = self.prompt_template\
                            .replace('<<EVIDENCE>>', this_doc['text'])\
                            .replace('<<QUESTION>>', question)
            
            batch_texts.append(query_text)
            batch_imgs.append(query_img)
            batch_docs.append(this_doc)

            # Process the batch if it's full
            if len(batch_texts) == self.bz:
                batch_scores = self.get_next_token_ps(batch_texts, batch_imgs)
                for doc, score in zip(batch_docs, batch_scores):
                    doc['rerank_score'] = score
                    all_scores.append(score)

                # Reset the batch
                batch_texts = []
                batch_imgs = []
                batch_docs = []

        # Process any remaining documents in the final batch
        if batch_texts:
            batch_scores = self.get_next_token_ps(batch_texts, batch_imgs)
            for doc, score in zip(batch_docs, batch_scores):
                doc['rerank_score'] = score
                all_scores.append(score)

        reranked_documents = sorted(retrieved_docs, key=lambda x: x['rerank_score'], reverse=True)
        return reranked_documents
        

        
        

class Classifier(nn.Module):
    def __init__(self, input_shape, num_layers, proj_dim, output_dim=1, input_dropout=0., dropout=None):
        super(Classifier, self).__init__()
        layers = []

        
        
        # Handle dropout initialization
        if dropout is None:
            dropout = [0.0] * num_layers

        # Input dropout layer
        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        
        
        
        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_shape, proj_dim))
            #nn.init.xavier_uniform_(layers[-1].weight)  # Xavier initialization
            layers.append(nn.ReLU())
            if dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
            input_shape = proj_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(proj_dim, output_dim)
        #nn.init.xavier_uniform_(self.output_layer.weight)  # Initialize output layer as well

    def forward(self, x, return_embed=False):
        embed = self.mlp(x)
        output = self.output_layer(embed)
        if return_embed:
            return output, embed
        return output