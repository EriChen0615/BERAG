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
        ops.Retrieve_OP {
            kwargs: {
                ret_topk: 10,
                use_cache: true,
                cache_file: "cache/EVQA/retrieve/Retrieve_histories.json"
            }
        },
        ops.VLMRead_OP,
    ],
}