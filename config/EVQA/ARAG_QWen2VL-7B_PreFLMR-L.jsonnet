local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";

{
    agent_config:  {
        class_name: 'RAG_VQA_Agent', 
        vlm_class: 'QWen2VLM',
        vlm_config: models.QWen2VL_7B_Config,
        retriever_class: 'FLMRRetriever', 
        retriever_config: retrievers.EVQA_PreFLMRRetriever_ViT_L,
    },
    initial_prompt: "config/prompts/1006_arag.txt"
}