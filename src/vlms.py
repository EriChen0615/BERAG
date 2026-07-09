from abc import ABC, abstractmethod
import torch
from transformers import DynamicCache, OffloadedCache
import copy
import base64
import requests
from io import BytesIO
from PIL import Image
import openai
import os

def are_model_weights_identical(model1, model2):
    # Iterate through parameters in both models
    identical = True
    for (name_self, param_self), (name_base, param_base) in zip(model1.named_parameters(), model2.named_parameters()):
        if name_self != name_base:
            print(f"Layer names differ: {name_self} vs {name_base}")
            identical = False
            break
        if not torch.allclose(param_self, param_base, atol=1e-6):
            print(f"Difference found in layer: {name_self}")
            identical = False
            break

    if identical:
        print("All weights are identical!")
    else:
        print("Weights differ between models.")
    return identical
    
class All_VLM(ABC):
    def __init__(self, model_path, generation_config, *args, is_lora=False, base_model_path=None, processor_path=None, attn_implementation="flash_attention_", **kwargs):
        self.model_path = model_path
        self.generation_config = generation_config
        self.extra_kwargs = kwargs
        self.is_lora = is_lora
        self.base_model_path = base_model_path
        self.processor_path = processor_path
        self.attn_implementation = attn_implementation

    @abstractmethod
    def generate_response(self, input_text, input_img, seed):
        pass

class OpenAI_VLM(All_VLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from openai import OpenAI  # Updated import
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
        
    def _encode_image(self, image_path):
        # Read and encode local image file
        with open(image_path, 'rb') as image_file:
            return image_file.read()

    @torch.no_grad()
    def generate_response(self, input_text, input_img, history=None, seed=0, return_logps=False, return_many_seqs=False, num_return_sequences=None):
        if return_logps:
            raise NotImplementedError("return_logps not supported for OpenAI models")
            
        try:
            # Read the image file
            image_data = self._encode_image(input_img)

            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url":  f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"
                                }  # Can be a file path or URL
                            },
                            {
                                "type": "text",
                                "text": input_text
                            }
                        ]
                    }
                ],
                max_tokens=self.generation_config.get("max_new_tokens", 300),
                temperature=self.generation_config.get("temperature", 0),
                n=self.generation_config.get("num_return_sequences", 1) if return_many_seqs else 1
            )
            
            if return_many_seqs:
                return [choice.message.content for choice in response.choices]
            else:
                return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return "[ERROR] Failed to generate response from OpenAI API"


class DummyVLM(All_VLM):
    def __init__(self, always_return, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.always_return = always_return

    def generate_response(self, input_text, input_img):
        return self.always_return + input_text + input_img


from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
class QWen2VLM(All_VLM):
    def _init_model(self):
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        if not self.is_lora:
            print(f"Initialize VLM from {self.model_path}")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                # attn_implementation"eager",
                attn_implementation=self.attn_implementation,
                device_map="auto",
                **self.extra_kwargs,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(self.processor_path or self.model_path)
        else:
            print(f"Initialize VLM (LoRA). Base model = {self.base_model_path}; LoRA = {self.model_path}")
            assert self.base_model_path is not None
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
                attn_implementation=self.attn_implementation,
                device_map="auto",
                **self.extra_kwargs,
            ).eval()
            self.model.load_adapter(self.model_path)
            self.processor = AutoProcessor.from_pretrained(self.processor_path or self.base_model_path)

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None # lazy initialization
        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.

    @torch.no_grad()
    def generate_response(self, input_text, input_img, history=None, seed=0, return_logps=False, return_many_seqs=False):
        if self.model is None:
            self._init_model()
        torch.manual_seed(seed)
        # print(f"Seed set to {seed}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_img},
                    {"type": "text", "text": input_text},
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
        # if len(inputs["input_ids"][0]) > 4096:
        #     return "\n[ANSWER] Fail to generate answer (context exceeded 4096)"

        inputs = inputs.to("cuda")

        # Inference
        # print("Generation config=", self.generation_config)
        if return_logps is False:
            oom = False
            try:
                generated_ids = self.model.generate(**inputs, **self.generation_config)
            except torch.cuda.OutOfMemoryError as e:
                print(e)
                print("retrying with cache_implementation=offloaded")
                oom = True
            if oom:
                torch.cuda.empty_cache()
                new_generation_config = copy.deepcopy(self.generation_config)
                new_generation_config['cache_implementation'] = "offloaded"
                # offloaded_cache = OffloadedCache()
                generated_ids = self.model.generate(**inputs, **new_generation_config)
        else:
            output_dict = self.model.generate(**inputs, **self.generation_config, return_dict_in_generate=True, output_scores=True)
            generated_ids = output_dict['sequences']
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if return_logps is False:
            return output_text[0] if return_many_seqs == False else output_text
        else:
            past_key_values = output_dict['past_key_values']
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            #DEBUG 
            context_length = inputs['input_ids'].size(1) 
            total_length = past_key_values.get_seq_length()
            del inputs

            new_inputs = self.processor(
                text=[text+output_text[0]],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to('cuda')  
            forward_outputs = self.model(**new_inputs, past_key_values=past_key_values, use_cache=True)

            # Extract logits from forward_outputs
            logits = forward_outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

            # Compute log probabilities using log_softmax
            log_probs = torch.log_softmax(logits, dim=-1)  # Apply softmax on the last dimension (vocab size)
            # Initialize list to store log-probs for each sequence in the batch
            batch_log_probs = []
            batch_gen_length = []

            # Loop over each sequence in the batch
            for i in range(generated_ids.size(0)):  # Batch size dimension
                sequence_log_probs = []

                # Loop over each token in the sequence
                for j in range(context_length, total_length):  # Sequence length dimension
                    token_id = generated_ids[i, j].item()  # Get the token id at this position
                    log_prob_token = log_probs[i, j-1, token_id].item()  # Get the log-prob of that token
                    
                    # Store the log-prob of the token
                    sequence_log_probs.append(log_prob_token)

                # Sum the log-probs to get the total log-probability of the sequence
                total_log_prob_sequence = sum(sequence_log_probs)
                
                # Store the total log-prob of the sequence
                batch_log_probs.append(total_log_prob_sequence)
                batch_gen_length.append(total_length - context_length)

            return output_text[0], batch_log_probs[0], batch_gen_length[0]
        
    @torch.no_grad()
    def generate_multiple_responses(self, input_text, input_img, history=None, seed=0, num_return_sequences=None):
        if self.model is None:
            self._init_model()
        torch.manual_seed(seed)
        # print(f"Seed set to {seed}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": input_img},
                    {"type": "text", "text": input_text},
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
        # if len(inputs["input_ids"][0]) > 4096:
        #     return "\n[ANSWER] Fail to generate answer (context exceeded 4096)"

        inputs = inputs.to("cuda")

        # Inference
        # print("Generation config=", self.generation_config)
        # oom = False
        # try:
        # breakpoint() #NOTE: check **self.generation_config
        if 'num_return_sequences' in self.generation_config:
            self.generation_config.pop('num_return_sequences') # do not use num_return_sequences in generation config
        generated_ids = self.model.generate(**inputs, **self.generation_config, num_return_sequences=num_return_sequences)
        # except torch.cuda.OutOfMemoryError as e:
        #     print(e)
        #     print("retrying with cache_implementation=offloaded")
        #     oom = True
        # if oom:
        #     torch.cuda.empty_cache()
        #     new_generation_config = copy.deepcopy(self.generation_config)
        #     new_generation_config['cache_implementation'] = "offloaded"
        #     # offloaded_cache = OffloadedCache()
        #     generated_ids = self.model.generate(**inputs, **new_generation_config)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids.repeat_interleave(num_return_sequences, dim=0), generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
    
    @torch.no_grad()
    def get_next_token_ps_from_model(self, y_nexts, context, img0, img1):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img0, "resized_width": 512, "resized_height": 512},
                    {"type": "image", "image": img1, "resized_width": 512, "resized_height": 512},
                    {"type": "text", "text": context},
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
        #NOTE for DEUBGGING
        # from pprint import pprint
        # most_likely_next_token_id = torch.argmax(last_probs)
        # most_likely_next_token = self.processor.batch_decode(
        #     [[most_likely_next_token_id]], skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        # pprint(output_probs)
        # print(most_likely_next_token)
        # breakpoint()
        return output_probs
