local models = import "../models.libsonnet";

{
    agent_config:  {
        class_name: 'Plain_VQA_Agent', 
        vlm_class: 'QWen2VLM',
        vlm_config: models.QWen2VL_7B_4bit_Config,
    },
    initial_prompt: "config/prompts/1008_short_norag.txt"
}