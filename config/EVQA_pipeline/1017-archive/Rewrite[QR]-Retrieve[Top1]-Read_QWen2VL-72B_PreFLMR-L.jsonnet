local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";
local ops = import "../operators.libsonnet";

{
    agent_config:  {
        vlm_class: 'QWen2VLM',
        vlm_config: models.QWen2VL_72B_Config,
        retriever_class: 'FLMRRetriever', 
        retriever_config: retrievers.EVQA_PreFLMRRetriever_ViT_L,
    },
    initial_prompt: "config/prompts/1003_conventional_rag.txt",
    op_config: [
        ops.Rewrite_QR_OP,
        ops.Retrieve_OP,
        ops.VLMRead_OP,
    ],
}