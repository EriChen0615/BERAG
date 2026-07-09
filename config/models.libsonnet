local QWen2VL_7B_Config = {
    'model_path': 'QWen/QWen2-VL-7B-Instruct',
    'processor_path': 'QWen/QWen2-VL-7B-Instruct', 
    'generation_config': {
        'temperature': 0.3, 
        'max_new_tokens': 64
    }
};

local QWen2VL_7B_LoRA_Config = {
    model_path: 'third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/rag1_answer/checkpoint-5000',
    generation_config: {
        temperature: 0.3, 
        max_new_tokens: 64
    },
    is_lora: true,
    base_model_path: 'QWen/QWen2-VL-7B-Instruct',
    processor_path: 'QWen/QWen2-VL-7B-Instruct', 
};


local QWen2VL_7B_4bitGPTQ_Config = {
    'model_path': 'Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4',
    'generation_config': {
        'temperature': 0.3, 
        'max_new_tokens': 512
    }
};

local QWen2VL_7B_8bitGPTQ_Config = {
    'model_path': 'QWen/QWen2-VL-7B-Instruct-GPTQ-Int8',
    'generation_config': {
        'temperature': 0.3, 
        'max_new_tokens': 512
    }
};

local QWen2VL_7B_4bit_Config = {
    'model_path': 'Qwen/Qwen2-VL-7B-Instruct',
    'generation_config': {
        'temperature': 0.3, 
        'max_new_tokens': 512
    },
    'load_in_4bit': true
};

local QWen2VL_7B_8bit_Config = {
    'model_path': 'QWen/QWen2-VL-7B-Instruct-GPTQ',
    'generation_config': {
        'temperature': 0.3, 
        'max_new_tokens': 512
    },
    'load_in_8bit': true

};

local QWen2VL_72B_Config = {
    'model_path': '../HF_HOME/QWen2-VL-72B-Instruct',
    'generation_config': {
        'temperature': 0.3,
        'max_new_tokens': 512
    }
};

local QWen2VL_72B_8bitGPTQ_Config = {
    'model_path': '../HF_HOME/QWen2-VL-72B-Instruct-GPTQ-Int8',
    'generation_config': {
        'temperature': 0.3,
        'max_new_tokens': 512
    }
};

local QWen2VL_72B_4bitGPTQ_Config = {
    'model_path': '../HF_HOME/QWen2-VL-72B-Instruct-GPTQ-Int4',
    'generation_config': {
        'temperature': 0.3,
        'max_new_tokens': 512
    }
};

local QWen2VL_2B_Config = {
    'model_path': 'QWen/QWen2-VL-2B-Instruct',
    'processor_path': 'QWen/QWen2-VL-2B-Instruct',
    'generation_config': {
        'temperature': 0.3, 
        'max_new_tokens': 64
    }
};

local QWen2VL_2B_LoRA_Config = {
    model_path: 'third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/sft-rewrite_doc_gen/checkpoint-4218',
    generation_config: {
        temperature: 0.3, 
        max_new_tokens: 64
    },
    is_lora: true,
    base_model_path: 'QWen/QWen2-VL-2B-Instruct',
    processor_path: 'QWen/QWen2-VL-2B-Instruct',
};

local QWen2VL_2B_LoRA_NoRAGRead_Config = {
    model_path: 'third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/norag_answer/checkpoint-5225',
    generation_config: {
        temperature: 0.3, 
        max_new_tokens: 64
    },
    is_lora: true,
    base_model_path: 'QWen/QWen2-VL-2B-Instruct'
};

local GPT4omini_Config = {
    model_path: 'gpt-4o-mini-2024-07-18',
    // model_path: 'gpt-4o-mini',
    generation_config: {
        temperature: 0.3, 
        max_new_tokens: 64
    }
};  

{
    QWen2VL_2B_Config: QWen2VL_2B_Config,
    QWen2VL_2B_LoRA_Config: QWen2VL_2B_LoRA_Config,
    QWen2VL_2B_LoRA_NoRAGRead_Config: QWen2VL_2B_LoRA_NoRAGRead_Config,
    QWen2VL_7B_Config: QWen2VL_7B_Config,
    QWen2VL_7B_LoRA_Config: QWen2VL_7B_LoRA_Config,
    QWen2VL_7B_4bit_Config: QWen2VL_7B_4bit_Config,
    QWen2VL_7B_4bitGPTQ_Config: QWen2VL_7B_4bitGPTQ_Config,
    QWen2VL_7B_8bit_Config: QWen2VL_7B_8bit_Config,
    QWen2VL_7B_8bitGPTQ_Config: QWen2VL_7B_8bitGPTQ_Config,
    QWen2VL_72B_Config: QWen2VL_72B_Config,
    QWen2VL_72B_8bitGPTQ_Config: QWen2VL_72B_8bitGPTQ_Config,
    QWen2VL_72B_4bitGPTQ_Config: QWen2VL_72B_4bitGPTQ_Config,
    GPT4omini_Config: GPT4omini_Config,
}