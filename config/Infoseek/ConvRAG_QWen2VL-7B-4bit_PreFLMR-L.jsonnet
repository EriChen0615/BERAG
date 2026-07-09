local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";

{
    agent_config:  {
        class_name: 'ConventionalRAG_VQA_Agent', 
        vlm_class: 'QWen2VLM',
        vlm_config: models.QWen2VL_7B_4bit_Config,
        retriever_class: 'FLMRRetriever', 
        retriever_config: retrievers.Infoseek_PreFLMRRetriever_ViT_L,
    },
    initial_prompt: "config/prompts/1008_conventional_rag_short.txt"
}