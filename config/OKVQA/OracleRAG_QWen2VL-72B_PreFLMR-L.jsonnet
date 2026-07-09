local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";

{
    agent_config:  {
        class_name: 'ConventionalRAG_VQA_Agent', 
        vlm_class: 'QWen2VLM',
        vlm_config: models.QWen2VL_72B_Config,
        retriever_class: 'FLMRRetriever', 
        retriever_config: retrievers.OKVQA_OrcaleRetriever,
    },
    initial_prompt: "config/prompts/1003_conventional_rag.txt"
}