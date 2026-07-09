local models = import "../models.libsonnet";
local retrievers = import "../retrievers.libsonnet";
local ops = import "../operators.libsonnet";
local rerankers = import "../rerankers.libsonnet";

{
    agent_config:  {
        vlm_class: 'QWen2VLM',
        vlm_config: models.QWen2VL_7B_LoRA_Config,
        retriever_class: 'FLMRRetriever', 
        retriever_config: retrievers.EVQA_PreFLMRRetriever_ViT_L {
            use_gpu: true
            deduplicate: true,
        },
    },
    initial_prompt: "config/prompts/1003_conventional_rag.txt",
    op_config: [
        ops.Retrieve_OP {
            name: "RetrieveRerank",
            kwargs: {
                ret_topk: 1,
                use_cache: true,
                cache_file: "cache/EVQA_test256/retrieve-rerank/RetrieveRerank_histories.json",
                reranker_class: "QWen2Reranker",
                reranker_config: rerankers.EVQA_QWen2VL_2B_LoRA_Config,
                rerank_topk: 50,
            }
        },
        ops.VLMRead_OP,
    ],
}