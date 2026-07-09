local models = import "../../models.libsonnet";
local retrievers = import "../../retrievers.libsonnet";
local ops = import "../../operators.libsonnet";
local rerankers = import "../../rerankers.libsonnet";

{
    agent_config:  {
        vlm_class: 'QWen2VLM',
        vlm_config: models.QWen2VL_7B_Config,
        retriever_class: 'CacheDatasetRetriever', 
        retriever_config: retrievers.EVQA_test_CacheDatasetRetriever_ViT_L {
            retrieval_field: "reranked_passage",
        },
    },
    initial_prompt: "config/prompts/1003_conventional_rag.txt",
    op_config: [
        ops.Retrieve_OP {
            kwargs: {
                ret_topk: 1,
                use_cache: false,
                cache_file: "",
            }
        },
        ops.VLMRead_OP,
    ],
}