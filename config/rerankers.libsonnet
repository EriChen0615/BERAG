local EVQA_QWen2VL_2B_LoRA_Config = {
    model_path: 'third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/doc1_verify/checkpoint-10450',
    is_lora: true,
    base_model_path: 'QWen/QWen2-VL-2B-Instruct',
    processor_path: 'QWen/QWen2-VL-2B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};

local EVQA_QWen2VL_7B_LoRA_Config = {
    model_path: 'third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/doc1_verify/checkpoint-10450',
    is_lora: true,
    base_model_path: 'QWen/QWen2-VL-7B-Instruct',
    processor_path: 'QWen/QWen2-VL-7B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};

local EVQA_QWen2VLCLS_2B_LoRA_Config = {
    model_path: '/home/jm2245/rds/rds-cvnlp-hirYTW1FQIw/shared_space/jm2245/LAMAFACT-MMHS/checkpoints/qwen2_vl-2b/qlora/evqa/2024-12-28_doc1_verify',
    is_lora: true,
    is_cls: true,
    base_model_path: 'QWen/QWen2-VL-2B-Instruct',
    processor_path: 'QWen/QWen2-VL-2B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};
local EVQA_QWen2VLCLS_7B_LoRA_Config = {
    model_path: '/home/jm2245/rds/rds-cvnlp-hirYTW1FQIw/shared_space/jm2245/LAMAFACT-MMHS/checkpoints/qwen2_vl-7b/qlora/evqa/2024-12-28_doc1_verify',
    is_lora: true,
    is_cls: true,
    base_model_path: 'QWen/QWen2-VL-7B-Instruct',
    processor_path: 'QWen/QWen2-VL-7B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};


local OKVQA_QWen2VL_2B_LoRA_Config = {
    model_path: 'third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/okvqa/doc1_verify/checkpoint-1686',
    is_lora: true,
    base_model_path: 'QWen/QWen2-VL-2B-Instruct',
    processor_path: 'QWen/QWen2-VL-2B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};

local OKVQA_QWen2VL_7B_LoRA_Config = {
    model_path: 'third_party/LLaMAFactory/saves/qwen2_vl-7b/lora/okvqa/doc1_verify/checkpoint-1686',
    is_lora: true,
    base_model_path: 'QWen/QWen2-VL-7B-Instruct',
    processor_path: 'QWen/QWen2-VL-7B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};


local OKVQA_QWen2VLCLS_2B_LoRA_Config = {
    model_path: '/home/jm2245/rds/rds-cvnlp-hirYTW1FQIw/shared_space/jm2245/LAMAFACT-MMHS/checkpoints/qwen2_vl-2b/qlora/okvqa/2024-12-28_doc1_verify',
    is_lora: true,
    is_cls: true,
    base_model_path: 'QWen/QWen2-VL-2B-Instruct',
    processor_path: 'QWen/QWen2-VL-2B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};

local OKVQA_QWen2VLCLS_7B_LoRA_Config = {
    model_path: '/home/jm2245/rds/rds-cvnlp-hirYTW1FQIw/shared_space/jm2245/LAMAFACT-MMHS/checkpoints/qwen2_vl-7b/qlora/okvqa/2024-12-28_doc1_verify',
    is_lora: true,
    is_cls: true,
    base_model_path: 'QWen/QWen2-VL-7B-Instruct',
    processor_path: 'QWen/QWen2-VL-7B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};


local Infoseek_QWen2VL_2B_LoRA_Config = {
    model_path: 'third_party/LLaMAFactory/saves/qwen2_vl-2b/lora/infoseek/doc1_verify/checkpoint-2000',
    is_lora: true,
    base_model_path: 'QWen/QWen2-VL-2B-Instruct',
    processor_path: 'QWen/QWen2-VL-2B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};

local Infoseek_QWen2VLCLS_2B_LoRA_Config = {
    model_path: '/home/jm2245/rds/rds-cvnlp-hirYTW1FQIw/shared_space/jm2245/LAMAFACT-MMHS/checkpoints/qwen2_vl-2b/qlora/infoseek/2024-12-28_doc1_verify',
    is_lora: true,
    is_cls: true,
    base_model_path: 'QWen/QWen2-VL-2B-Instruct',
    processor_path: 'QWen/QWen2-VL-2B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};


local InfoseekNew_QWen2VLCLS_2B_LoRA_Config = {
    model_path: '/home/jm2245/rds/rds-cvnlp-hirYTW1FQIw/shared_space/jm2245/LAMAFACT-MMHS/checkpoints/qwen2_vl-2b/qlora/infoseeknew/2024-12-28_doc1_verify',
    is_lora: true,
    is_cls: true,
    base_model_path: 'QWen/QWen2-VL-2B-Instruct',
    processor_path: 'QWen/QWen2-VL-2B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};

local InfoseekNew_QWen2VLCLS_7B_LoRA_Config = {
    model_path: '/home/jm2245/rds/rds-cvnlp-hirYTW1FQIw/shared_space/jm2245/LAMAFACT-MMHS/checkpoints/qwen2_vl-7b/qlora/infoseeknew/2024-12-28_doc1_verify',
    is_lora: true,
    is_cls: true,
    base_model_path: 'QWen/QWen2-VL-7B-Instruct',
    processor_path: 'QWen/QWen2-VL-7B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};

local InfoseekNew_QWen2VL_2B_LoRA_Config = {
    model_path: '/home/jm2245/rds/rds-cvnlp-hirYTW1FQIw/shared_space/jm2245/LAMAFACT-MMHS/saves/qwen2_vl-2b/lora/infoseeknew/doc1_verify',
    is_lora: true,
    base_model_path: 'QWen/QWen2-VL-2B-Instruct',
    processor_path: 'QWen/QWen2-VL-2B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};


local InfoseekNew_QWen2VL_7B_LoRA_Config = {
    model_path: '/home/jm2245/rds/rds-cvnlp-hirYTW1FQIw/shared_space/jm2245/LAMAFACT-MMHS/saves/qwen2_vl-7b/lora/infoseeknew/doc1_verify',
    is_lora: true,
    base_model_path: 'QWen/QWen2-VL-7B-Instruct',
    processor_path: 'QWen/QWen2-VL-7B-Instruct',
    prompt_template_file: 'config/prompts/1111_doc1_verify.txt',
};



{
    EVQA_QWen2VL_2B_LoRA_Config: EVQA_QWen2VL_2B_LoRA_Config,
    EVQA_QWen2VL_7B_LoRA_Config: EVQA_QWen2VL_7B_LoRA_Config,
    OKVQA_QWen2VL_2B_LoRA_Config: OKVQA_QWen2VL_2B_LoRA_Config,
    OKVQA_QWen2VL_7B_LoRA_Config: OKVQA_QWen2VL_7B_LoRA_Config,
    Infoseek_QWen2VL_2B_LoRA_Config: Infoseek_QWen2VL_2B_LoRA_Config,
    InfoseekNew_QWen2VL_2B_LoRA_Config: InfoseekNew_QWen2VL_2B_LoRA_Config,
    InfoseekNew_QWen2VL_7B_LoRA_Config: InfoseekNew_QWen2VL_7B_LoRA_Config,
    EVQA_QWen2VLCLS_2B_LoRA_Config: EVQA_QWen2VLCLS_2B_LoRA_Config,
    EVQA_QWen2VLCLS_7B_LoRA_Config: EVQA_QWen2VLCLS_7B_LoRA_Config,
    OKVQA_QWen2VLCLS_2B_LoRA_Config: OKVQA_QWen2VLCLS_2B_LoRA_Config,
    OKVQA_QWen2VLCLS_7B_LoRA_Config: OKVQA_QWen2VLCLS_7B_LoRA_Config,
    Infoseek_QWen2VLCLS_2B_LoRA_Config: Infoseek_QWen2VLCLS_2B_LoRA_Config,
    InfoseekNew_QWen2VLCLS_2B_LoRA_Config: InfoseekNew_QWen2VLCLS_2B_LoRA_Config,
    InfoseekNew_QWen2VLCLS_7B_LoRA_Config: InfoseekNew_QWen2VLCLS_7B_LoRA_Config,
}